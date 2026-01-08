"""
Fila de Treinamento
===================
Gerencia fila de treinamento de modelos em background.
Persiste estado em arquivo JSON para sobreviver a restarts.

FIAP - Tech Challenge Fase 4
"""

import os
import json
import logging
import threading
import queue
from datetime import datetime
from typing import Optional, Dict, List, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import traceback

from app.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingStatus(str, Enum):
    """Status possíveis de um treinamento."""
    QUEUED = "queued"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Representa um job de treinamento."""
    symbol: str
    status: TrainingStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: int = 0
    epochs_total: int = 100
    epochs_current: int = 0
    error_message: Optional[str] = None
    parameters: Optional[Dict] = None
    result_metrics: Optional[Dict] = None
    
    def to_dict(self) -> dict:
        """Converte para dicionário."""
        d = asdict(self)
        d["status"] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrainingJob":
        """Cria instância a partir de dicionário."""
        data["status"] = TrainingStatus(data["status"])
        return cls(**data)


class TrainingQueue:
    """
    Fila de treinamento com worker em background.
    
    - Executa 1 treinamento por vez
    - Persiste estado em arquivo JSON
    - Thread-safe
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.settings = get_settings()
        self._queue: queue.Queue = queue.Queue()
        self._jobs: Dict[str, TrainingJob] = {}
        self._current_job: Optional[str] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._jobs_lock = threading.Lock()
        
        # Caminho para persistir fila
        self._queue_file = os.path.join(self.settings.models_dir, "training_queue.json")
        
        # Carrega estado persistido
        self._load_state()
        
        self._initialized = True
    
    def _load_state(self):
        """Carrega estado da fila do arquivo JSON."""
        if os.path.exists(self._queue_file):
            try:
                with open(self._queue_file, 'r') as f:
                    data = json.load(f)
                
                for job_data in data.get("jobs", []):
                    job = TrainingJob.from_dict(job_data)
                    self._jobs[job.symbol] = job
                    
                    # Re-enfileira jobs que estavam pendentes
                    if job.status in [TrainingStatus.QUEUED, TrainingStatus.TRAINING]:
                        job.status = TrainingStatus.QUEUED
                        job.progress = 0
                        job.started_at = None
                        self._queue.put(job.symbol)
                
                logger.info(f"Fila carregada: {len(self._jobs)} jobs")
            except Exception as e:
                logger.error(f"Erro ao carregar fila: {e}")
    
    def _save_state(self):
        """Persiste estado da fila em arquivo JSON."""
        try:
            os.makedirs(os.path.dirname(self._queue_file), exist_ok=True)
            
            with self._jobs_lock:
                data = {
                    "updated_at": datetime.now().isoformat(),
                    "jobs": [job.to_dict() for job in self._jobs.values()]
                }
            
            with open(self._queue_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Erro ao salvar fila: {e}")
    
    def start_worker(self):
        """Inicia worker thread para processar fila."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.info("Worker já está rodando")
            return
        
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="TrainingWorker"
        )
        self._worker_thread.start()
        logger.info("Worker de treinamento iniciado")
    
    def stop_worker(self):
        """Para worker thread."""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5)
        logger.info("Worker de treinamento parado")
    
    def _worker_loop(self):
        """Loop principal do worker."""
        logger.info("Worker loop iniciado")
        
        while not self._stop_event.is_set():
            try:
                # Espera por job na fila (timeout para verificar stop_event)
                try:
                    symbol = self._queue.get(timeout=2)
                except queue.Empty:
                    continue
                
                # Processa job
                self._process_job(symbol)
                self._queue.task_done()
                
            except Exception as e:
                logger.error(f"Erro no worker loop: {e}")
                traceback.print_exc()
        
        logger.info("Worker loop finalizado")
    
    def _process_job(self, symbol: str):
        """Processa um job de treinamento."""
        logger.info(f"[{symbol}] Iniciando treinamento...")
        
        with self._jobs_lock:
            if symbol not in self._jobs:
                return
            
            job = self._jobs[symbol]
            job.status = TrainingStatus.TRAINING
            job.started_at = datetime.now().isoformat()
            self._current_job = symbol
        
        self._save_state()
        
        try:
            # Importa aqui para evitar circular imports
            from model.train import train_model
            from app.core.model_registry import get_model_registry
            
            # Parâmetros do treinamento
            params = job.parameters or {}
            epochs = params.get("epochs", 100)
            start_date = params.get("start_date", "2018-01-01")
            batch_size = params.get("batch_size", 32)
            
            # Diretório de saída (base - train.py vai criar subdiretório com símbolo)
            model_dir = self.settings.models_dir
            
            # Callback de progresso
            def progress_callback(epoch: int, total: int):
                with self._jobs_lock:
                    if symbol in self._jobs:
                        self._jobs[symbol].epochs_current = epoch
                        self._jobs[symbol].epochs_total = total
                        self._jobs[symbol].progress = int((epoch / total) * 100)
                self._save_state()
            
            # Executa treinamento
            results = train_model(
                symbol=symbol,
                start_date=start_date,
                epochs=epochs,
                batch_size=batch_size,
                model_dir=model_dir,
                progress_callback=progress_callback
            )
            
            # Atualiza job com sucesso
            with self._jobs_lock:
                job.status = TrainingStatus.COMPLETED
                job.completed_at = datetime.now().isoformat()
                job.progress = 100
                job.result_metrics = results.get("evaluation_metrics")
            
            # Registra no ModelRegistry
            registry = get_model_registry()
            registry.register_model(
                symbol=symbol,
                status="ready",
                trained_at=datetime.now().isoformat(),
                metrics=results.get("evaluation_metrics")
            )
            
            logger.info(f"[{symbol}] ✓ Treinamento concluído com sucesso!")
            
        except Exception as e:
            logger.error(f"[{symbol}] ✗ Erro no treinamento: {e}")
            traceback.print_exc()
            
            with self._jobs_lock:
                job.status = TrainingStatus.FAILED
                job.completed_at = datetime.now().isoformat()
                job.error_message = str(e)
        
        finally:
            self._current_job = None
            self._save_state()
    
    def enqueue(
        self,
        symbol: str,
        epochs: int = 100,
        start_date: str = "2018-01-01",
        batch_size: int = 32,
        **kwargs
    ) -> TrainingJob:
        """
        Adiciona símbolo à fila de treinamento.
        
        Args:
            symbol: Símbolo da ação
            epochs: Número de épocas
            start_date: Data inicial para dados
            batch_size: Tamanho do batch
            
        Returns:
            TrainingJob criado
        """
        symbol = symbol.upper()
        
        with self._jobs_lock:
            # Verifica se já está na fila ou em treinamento
            if symbol in self._jobs:
                existing = self._jobs[symbol]
                if existing.status in [TrainingStatus.QUEUED, TrainingStatus.TRAINING]:
                    logger.info(f"[{symbol}] Já está na fila/em treinamento")
                    return existing
            
            # Cria novo job
            job = TrainingJob(
                symbol=symbol,
                status=TrainingStatus.QUEUED,
                created_at=datetime.now().isoformat(),
                epochs_total=epochs,
                parameters={
                    "epochs": epochs,
                    "start_date": start_date,
                    "batch_size": batch_size,
                    **kwargs
                }
            )
            
            self._jobs[symbol] = job
        
        # Adiciona à fila
        self._queue.put(symbol)
        self._save_state()
        
        logger.info(f"[{symbol}] Adicionado à fila de treinamento")
        return job
    
    def get_job(self, symbol: str) -> Optional[TrainingJob]:
        """Retorna job por símbolo."""
        symbol = symbol.upper()
        with self._jobs_lock:
            return self._jobs.get(symbol)
    
    def get_job_status(self, symbol: str) -> Optional[dict]:
        """Retorna status de um job."""
        job = self.get_job(symbol)
        if job:
            return job.to_dict()
        return None
    
    def cancel_job(self, symbol: str) -> bool:
        """Cancela job na fila (não cancela se já em execução)."""
        symbol = symbol.upper()
        
        with self._jobs_lock:
            if symbol not in self._jobs:
                return False
            
            job = self._jobs[symbol]
            if job.status == TrainingStatus.QUEUED:
                job.status = TrainingStatus.CANCELLED
                job.completed_at = datetime.now().isoformat()
                self._save_state()
                logger.info(f"[{symbol}] Job cancelado")
                return True
            
            return False
    
    def remove_job(self, symbol: str) -> bool:
        """Remove job da lista (apenas concluídos/falhados/cancelados)."""
        symbol = symbol.upper()
        
        with self._jobs_lock:
            if symbol not in self._jobs:
                return False
            
            job = self._jobs[symbol]
            if job.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
                del self._jobs[symbol]
                self._save_state()
                return True
            
            return False
    
    def list_jobs(self) -> List[dict]:
        """Lista todos os jobs."""
        with self._jobs_lock:
            return [job.to_dict() for job in self._jobs.values()]
    
    def get_queue_status(self) -> dict:
        """Retorna status geral da fila."""
        with self._jobs_lock:
            queued = [j for j in self._jobs.values() if j.status == TrainingStatus.QUEUED]
            training = [j for j in self._jobs.values() if j.status == TrainingStatus.TRAINING]
            completed = [j for j in self._jobs.values() if j.status == TrainingStatus.COMPLETED]
            failed = [j for j in self._jobs.values() if j.status == TrainingStatus.FAILED]
            
            return {
                "worker_running": self._worker_thread is not None and self._worker_thread.is_alive(),
                "current_job": self._current_job,
                "queued_count": len(queued),
                "training_count": len(training),
                "completed_count": len(completed),
                "failed_count": len(failed),
                "queued_symbols": [j.symbol for j in queued],
                "training_symbol": training[0].symbol if training else None
            }
    
    def is_worker_running(self) -> bool:
        """Verifica se worker está rodando."""
        return self._worker_thread is not None and self._worker_thread.is_alive()


# Singleton
_training_queue: Optional[TrainingQueue] = None


def get_training_queue() -> TrainingQueue:
    """Retorna instância da fila de treinamento."""
    global _training_queue
    if _training_queue is None:
        _training_queue = TrainingQueue()
    return _training_queue


def init_training_queue() -> TrainingQueue:
    """Inicializa e inicia fila de treinamento."""
    queue = get_training_queue()
    queue.start_worker()
    return queue
