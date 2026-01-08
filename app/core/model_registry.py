"""
Registro de Modelos
===================
Gerencia múltiplos modelos LSTM, um por símbolo de ação.
Carrega/descarrega modelos sob demanda com cache.

FIAP - Tech Challenge Fase 4
"""

import os
import json
import logging
from typing import Optional, Dict, List
from datetime import datetime
from threading import Lock

import numpy as np
from tensorflow.keras.models import load_model
import joblib

from app.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduz logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ModelEntry:
    """
    Representa um modelo carregado para um símbolo específico.
    """
    
    def __init__(self, symbol: str, model_dir: str):
        self.symbol = symbol
        self.model_dir = model_dir
        self.model = None
        self.scaler_features = None
        self.scaler_target = None
        self.feature_columns = None
        self.training_results = None
        self.is_loaded = False
        self.loaded_at: Optional[datetime] = None
    
    def load(self) -> bool:
        """Carrega o modelo e seus artefatos."""
        try:
            # Carrega modelo
            model_path = os.path.join(self.model_dir, "lstm_model.keras")
            if not os.path.exists(model_path):
                logger.error(f"[{self.symbol}] Modelo não encontrado: {model_path}")
                return False
            
            logger.info(f"[{self.symbol}] Carregando modelo de: {model_path}")
            self.model = load_model(model_path)
            
            # Carrega scaler de features (opcional)
            scaler_features_path = os.path.join(self.model_dir, "scaler_features.pkl")
            if os.path.exists(scaler_features_path):
                self.scaler_features = joblib.load(scaler_features_path)
                logger.info(f"[{self.symbol}] Scaler de features carregado")
            
            # Carrega scaler do target
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            if not os.path.exists(scaler_path):
                logger.error(f"[{self.symbol}] Scaler não encontrado: {scaler_path}")
                return False
            
            self.scaler_target = joblib.load(scaler_path)
            logger.info(f"[{self.symbol}] Scaler de target carregado")
            
            # Carrega lista de features
            features_path = os.path.join(self.model_dir, "feature_columns.json")
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.feature_columns = json.load(f)
                logger.info(f"[{self.symbol}] {len(self.feature_columns)} features carregadas")
            else:
                self.feature_columns = [
                    'Close', 'RSI', 'MACD', 'MACD_Hist', 'BB_Position', 'BB_Width',
                    'ATR', 'Volatility', 'Momentum_10', 'Volume_Norm', 'ROC', 'Log_Return'
                ]
            
            # Carrega resultados do treinamento
            results_path = os.path.join(self.model_dir, "training_results.json")
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    self.training_results = json.load(f)
            
            self.is_loaded = True
            self.loaded_at = datetime.now()
            logger.info(f"[{self.symbol}] ✓ Modelo carregado com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"[{self.symbol}] Erro ao carregar modelo: {e}")
            self.is_loaded = False
            return False
    
    def unload(self):
        """Descarrega o modelo da memória."""
        self.model = None
        self.scaler_features = None
        self.scaler_target = None
        self.is_loaded = False
        self.loaded_at = None
        logger.info(f"[{self.symbol}] Modelo descarregado")
    
    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """Realiza predição."""
        if not self.is_loaded:
            raise RuntimeError(f"[{self.symbol}] Modelo não carregado")
        return self.model.predict(sequence, verbose=0)
    
    def scale_features(self, data: np.ndarray) -> np.ndarray:
        """Normaliza features."""
        if self.scaler_features is not None:
            return self.scaler_features.transform(data)
        return self.scaler_target.transform(data[:, 0].reshape(-1, 1))
    
    def scale(self, data: np.ndarray) -> np.ndarray:
        """Normaliza dados do target."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return self.scaler_target.transform(data)
    
    def inverse_scale(self, data: np.ndarray) -> np.ndarray:
        """Desnormaliza dados."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return self.scaler_target.inverse_transform(data)
    
    def get_n_features(self) -> int:
        """Retorna número de features."""
        return len(self.feature_columns) if self.feature_columns else 1
    
    def is_multivariate(self) -> bool:
        """Verifica se é multivariado."""
        return self.scaler_features is not None and self.get_n_features() > 1
    
    def get_info(self) -> dict:
        """Retorna informações do modelo."""
        info = {
            "symbol": self.symbol,
            "is_loaded": self.is_loaded,
            "is_multivariate": self.is_multivariate() if self.is_loaded else None,
            "n_features": self.get_n_features() if self.is_loaded else None,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None
        }
        
        if self.training_results:
            info["training_results"] = self.training_results
        
        return info


class ModelRegistry:
    """
    Registro central de modelos. Gerencia múltiplos modelos por símbolo.
    
    Estrutura esperada:
        models/
        ├── PETR4.SA/
        │   ├── lstm_model.keras
        │   ├── scaler.pkl
        │   ├── scaler_features.pkl
        │   ├── feature_columns.json
        │   └── training_results.json
        ├── VALE3.SA/
        │   └── ...
        └── registry.json  # Índice de modelos
    """
    
    _instance = None
    _lock = Lock()
    
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
        self.models_dir = self.settings.models_dir
        self._models: Dict[str, ModelEntry] = {}
        self._registry_path = os.path.join(self.models_dir, "registry.json")
        self._registry_data: dict = {}
        self._initialized = True
        
        # Carrega registro existente
        self._load_registry()
    
    def _load_registry(self):
        """Carrega o arquivo de registro."""
        if os.path.exists(self._registry_path):
            try:
                with open(self._registry_path, 'r') as f:
                    self._registry_data = json.load(f)
                logger.info(f"Registro carregado: {len(self._registry_data.get('models', {}))} modelos")
            except Exception as e:
                logger.error(f"Erro ao carregar registro: {e}")
                self._registry_data = {"models": {}}
        else:
            self._registry_data = {"models": {}}
            self._scan_models()
    
    def _save_registry(self):
        """Salva o arquivo de registro."""
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            with open(self._registry_path, 'w') as f:
                json.dump(self._registry_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erro ao salvar registro: {e}")
    
    def _scan_models(self):
        """Escaneia diretório de modelos e atualiza registro."""
        if not os.path.exists(self.models_dir):
            return
        
        for item in os.listdir(self.models_dir):
            item_path = os.path.join(self.models_dir, item)
            if os.path.isdir(item_path):
                model_file = os.path.join(item_path, "lstm_model.keras")
                if os.path.exists(model_file):
                    symbol = item
                    if symbol not in self._registry_data.get("models", {}):
                        # Carrega informações do treinamento
                        results_path = os.path.join(item_path, "training_results.json")
                        model_info = {
                            "status": "ready",
                            "created_at": datetime.now().isoformat(),
                            "model_dir": item_path
                        }
                        
                        if os.path.exists(results_path):
                            with open(results_path, 'r') as f:
                                results = json.load(f)
                                model_info["trained_at"] = results.get("training_end")
                                model_info["metrics"] = results.get("evaluation_metrics")
                        
                        if "models" not in self._registry_data:
                            self._registry_data["models"] = {}
                        self._registry_data["models"][symbol] = model_info
                        logger.info(f"Modelo encontrado: {symbol}")
        
        self._save_registry()
    
    def get_available_symbols(self) -> List[str]:
        """Retorna lista de símbolos com modelos disponíveis."""
        return [
            symbol for symbol, info in self._registry_data.get("models", {}).items()
            if info.get("status") == "ready"
        ]
    
    def has_model(self, symbol: str) -> bool:
        """Verifica se existe modelo para o símbolo."""
        symbol = symbol.upper()
        return (
            symbol in self._registry_data.get("models", {}) and
            self._registry_data["models"][symbol].get("status") == "ready"
        )
    
    def get_model(self, symbol: str) -> Optional[ModelEntry]:
        """
        Obtém modelo para um símbolo. Carrega sob demanda.
        
        Args:
            symbol: Símbolo da ação
            
        Returns:
            ModelEntry se disponível, None caso contrário
        """
        symbol = symbol.upper()
        
        # Verifica se modelo existe
        if not self.has_model(symbol):
            return None
        
        # Verifica se já está carregado
        if symbol in self._models and self._models[symbol].is_loaded:
            return self._models[symbol]
        
        # Carrega modelo
        model_dir = os.path.join(self.models_dir, symbol)
        entry = ModelEntry(symbol, model_dir)
        
        if entry.load():
            self._models[symbol] = entry
            return entry
        
        return None
    
    def get_model_info(self, symbol: str) -> Optional[dict]:
        """Retorna informações de um modelo específico."""
        symbol = symbol.upper()
        
        if symbol not in self._registry_data.get("models", {}):
            return None
        
        info = self._registry_data["models"][symbol].copy()
        info["symbol"] = symbol
        
        # Adiciona dados do training_results se existir
        model_dir = os.path.join(self.models_dir, symbol)
        results_path = os.path.join(model_dir, "training_results.json")
        
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                info["training_details"] = json.load(f)
        
        # Verifica se está carregado na memória
        info["is_loaded_in_memory"] = (
            symbol in self._models and self._models[symbol].is_loaded
        )
        
        return info
    
    def list_all_models(self) -> List[dict]:
        """Lista todos os modelos com status."""
        models = []
        for symbol, info in self._registry_data.get("models", {}).items():
            model_info = {
                "symbol": symbol,
                "status": info.get("status", "unknown"),
                "trained_at": info.get("trained_at"),
                "metrics": info.get("metrics"),
                "is_loaded_in_memory": (
                    symbol in self._models and self._models[symbol].is_loaded
                )
            }
            models.append(model_info)
        return models
    
    def register_model(self, symbol: str, status: str = "ready", **kwargs):
        """Registra ou atualiza um modelo no registro."""
        symbol = symbol.upper()
        
        if "models" not in self._registry_data:
            self._registry_data["models"] = {}
        
        model_info = {
            "status": status,
            "updated_at": datetime.now().isoformat(),
            **kwargs
        }
        
        if symbol in self._registry_data["models"]:
            self._registry_data["models"][symbol].update(model_info)
        else:
            model_info["created_at"] = datetime.now().isoformat()
            self._registry_data["models"][symbol] = model_info
        
        self._save_registry()
        logger.info(f"Modelo registrado: {symbol} - Status: {status}")
    
    def unregister_model(self, symbol: str):
        """Remove modelo do registro e descarrega da memória."""
        symbol = symbol.upper()
        
        if symbol in self._models:
            self._models[symbol].unload()
            del self._models[symbol]
        
        if symbol in self._registry_data.get("models", {}):
            del self._registry_data["models"][symbol]
            self._save_registry()
            logger.info(f"Modelo removido: {symbol}")
    
    def unload_model(self, symbol: str):
        """Descarrega modelo da memória (mantém no registro)."""
        symbol = symbol.upper()
        if symbol in self._models:
            self._models[symbol].unload()
            del self._models[symbol]
    
    def refresh(self):
        """Re-escaneia diretório e atualiza registro."""
        self._scan_models()
    
    def get_status(self) -> dict:
        """Retorna status geral do registro."""
        ready_models = [
            s for s, info in self._registry_data.get("models", {}).items()
            if info.get("status") == "ready"
        ]
        
        return {
            "total_registered": len(self._registry_data.get("models", {})),
            "ready_models": len(ready_models),
            "loaded_in_memory": len([m for m in self._models.values() if m.is_loaded]),
            "available_symbols": ready_models
        }


# Singleton
_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Retorna instância do registro de modelos."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry
