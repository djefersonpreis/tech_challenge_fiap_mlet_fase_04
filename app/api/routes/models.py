"""
Rotas de Gerenciamento de Modelos
=================================
CRUD de modelos e gerenciamento de fila de treinamento.

FIAP - Tech Challenge Fase 4
"""

import os
import shutil
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Path, Query

from app.api.schemas.models import (
    ModelListResponse,
    ModelDetailResponse,
    ModelSummary,
    ModelMetrics,
    ModelParameters,
    ModelDataInfo,
    TrainingRequest,
    TrainingJobResponse,
    TrainingQueueResponse,
    TrainingEnqueueResponse,
    ModelDeleteResponse,
    ErrorResponse
)
from app.core.model_registry import get_model_registry
from app.services.training_queue import get_training_queue, TrainingStatus
from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["Modelos"])


# ==========================================
# Fila de Treinamento (rotas sem parâmetros primeiro)
# ==========================================

@router.get(
    "/queue/status",
    response_model=TrainingQueueResponse,
    summary="Status da fila de treinamento",
    description="""
    Retorna status geral da fila de treinamento.
    
    Inclui:
    - Se o worker está ativo
    - Job atual em execução
    - Quantidade de jobs na fila
    - Lista de todos os jobs
    """
)
async def get_queue_status():
    """Retorna status da fila de treinamento."""
    training_queue = get_training_queue()
    
    status_data = training_queue.get_queue_status()
    jobs = training_queue.list_jobs()
    
    return TrainingQueueResponse(
        **status_data,
        jobs=[TrainingJobResponse(**j) for j in jobs]
    )


@router.post(
    "/registry/refresh",
    summary="Atualizar registro",
    description="Re-escaneia diretório de modelos e atualiza o registro."
)
async def refresh_registry():
    """Re-escaneia diretório e atualiza registro de modelos."""
    registry = get_model_registry()
    registry.refresh()
    
    return {
        "message": "Registro atualizado",
        "status": registry.get_status()
    }


# ==========================================
# Listar Modelos
# ==========================================

@router.get(
    "",
    response_model=ModelListResponse,
    summary="Listar todos os modelos",
    description="""
    Retorna lista de todos os modelos disponíveis e em treinamento.
    
    Inclui:
    - Modelos prontos para uso (ready)
    - Modelos em treinamento (training)
    - Modelos na fila (queued)
    - Modelos com falha (failed)
    """
)
async def list_models():
    """Lista todos os modelos registrados."""
    registry = get_model_registry()
    training_queue = get_training_queue()
    
    # Obtém modelos do registry
    models_list = registry.list_all_models()
    
    # Adiciona jobs da fila que ainda não estão no registry
    queue_jobs = training_queue.list_jobs()
    for job in queue_jobs:
        symbol = job["symbol"]
        # Verifica se já está na lista
        existing = next((m for m in models_list if m["symbol"] == symbol), None)
        
        if existing:
            # Atualiza status se está em treinamento
            if job["status"] in ["queued", "training"]:
                existing["status"] = job["status"]
        else:
            # Adiciona novo
            models_list.append({
                "symbol": symbol,
                "status": job["status"],
                "trained_at": job.get("completed_at"),
                "metrics": job.get("result_metrics"),
                "is_loaded_in_memory": False
            })
    
    # Converte para ModelSummary
    summaries = []
    for m in models_list:
        metrics = None
        if m.get("metrics"):
            metrics = ModelMetrics(**m["metrics"]) if isinstance(m["metrics"], dict) else None
        
        summaries.append(ModelSummary(
            symbol=m["symbol"],
            status=m["status"],
            trained_at=m.get("trained_at"),
            metrics=metrics,
            is_loaded_in_memory=m.get("is_loaded_in_memory", False)
        ))
    
    # Conta por status
    ready_count = len([s for s in summaries if s.status == "ready"])
    training_count = len([s for s in summaries if s.status in ["training", "queued"]])
    
    return ModelListResponse(
        models=summaries,
        total=len(summaries),
        ready_count=ready_count,
        training_count=training_count
    )


# ==========================================
# Detalhes de um Modelo
# ==========================================

@router.get(
    "/{symbol}",
    response_model=ModelDetailResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Modelo não encontrado"}
    },
    summary="Detalhes de um modelo",
    description="""
    Retorna informações detalhadas de um modelo específico.
    
    Inclui:
    - Métricas de avaliação (MAE, RMSE, MAPE, R²)
    - Parâmetros de treinamento
    - Informações dos dados utilizados
    - Features do modelo
    - Histórico de treinamento
    """
)
async def get_model_detail(
    symbol: str = Path(..., description="Símbolo da ação", examples=["PETR4.SA", "VALE3.SA"])
):
    """Retorna detalhes de um modelo específico."""
    symbol = symbol.upper()
    registry = get_model_registry()
    training_queue = get_training_queue()
    
    # Verifica no registry
    model_info = registry.get_model_info(symbol)
    
    # Verifica na fila de treinamento
    job = training_queue.get_job(symbol)
    
    if not model_info and not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modelo não encontrado para {symbol}"
        )
    
    # Monta resposta
    response_data = {
        "symbol": symbol,
        "status": "unknown",
        "is_loaded_in_memory": False
    }
    
    # Se tem info do registry
    if model_info:
        response_data["status"] = model_info.get("status", "ready")
        response_data["trained_at"] = model_info.get("trained_at")
        response_data["is_loaded_in_memory"] = model_info.get("is_loaded_in_memory", False)
        
        # Extrai detalhes do treinamento
        details = model_info.get("training_details", {})
        
        # Métricas
        if details.get("evaluation_metrics"):
            response_data["metrics"] = ModelMetrics(**details["evaluation_metrics"])
        elif model_info.get("metrics"):
            response_data["metrics"] = ModelMetrics(**model_info["metrics"])
        
        # Parâmetros
        if details.get("parameters"):
            params = details["parameters"]
            response_data["parameters"] = ModelParameters(
                sequence_length=params.get("sequence_length"),
                epochs_configured=params.get("epochs_configured") or params.get("epochs"),
                epochs_run=params.get("epochs_run"),
                batch_size=params.get("batch_size"),
                architecture=params.get("architecture"),
                lstm_units=params.get("lstm_units"),
                dropout_rate=params.get("dropout_rate"),
                learning_rate=params.get("learning_rate")
            )
        
        # Dados
        if details.get("data_info"):
            response_data["data_info"] = ModelDataInfo(**details["data_info"])
        
        # Features
        if details.get("features"):
            response_data["features"] = details["features"].get("columns", [])
            response_data["n_features"] = details["features"].get("n_features")
        
        # Histórico
        if details.get("training_history"):
            response_data["training_history"] = details["training_history"]
    
    # Se está na fila, atualiza status
    if job:
        if job.status in [TrainingStatus.QUEUED, TrainingStatus.TRAINING]:
            response_data["status"] = job.status.value
    
    return ModelDetailResponse(**response_data)


# ==========================================
# Solicitar Treinamento
# ==========================================

@router.post(
    "/{symbol}/train",
    response_model=TrainingEnqueueResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Parâmetros inválidos"},
        409: {"model": ErrorResponse, "description": "Já está em treinamento"}
    },
    summary="Solicitar treinamento de modelo",
    description="""
    Adiciona uma ação à fila de treinamento.
    
    O treinamento é executado em background por um worker dedicado.
    Apenas um treinamento é executado por vez.
    
    Use o endpoint GET /models/{symbol}/status para acompanhar o progresso.
    """
)
async def request_training(
    symbol: str = Path(..., description="Símbolo da ação", examples=["VALE3.SA", "ITUB4.SA"]),
    request: TrainingRequest = None
):
    """Adiciona símbolo à fila de treinamento."""
    symbol = symbol.upper()
    training_queue = get_training_queue()
    
    # Usa valores padrão se não fornecidos
    if request is None:
        request = TrainingRequest()
    
    # Verifica se já está em treinamento
    existing_job = training_queue.get_job(symbol)
    if existing_job and existing_job.status in [TrainingStatus.QUEUED, TrainingStatus.TRAINING]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{symbol} já está na fila ou em treinamento"
        )
    
    # Enfileira treinamento
    job = training_queue.enqueue(
        symbol=symbol,
        epochs=request.epochs,
        start_date=request.start_date,
        batch_size=request.batch_size
    )
    
    # Calcula posição na fila
    queue_status = training_queue.get_queue_status()
    position = queue_status["queued_count"]
    if queue_status["training_symbol"]:
        position += 1
    
    return TrainingEnqueueResponse(
        message=f"Treinamento de {symbol} adicionado à fila",
        job=TrainingJobResponse(**job.to_dict()),
        queue_position=position
    )


# ==========================================
# Status de Treinamento
# ==========================================

@router.get(
    "/{symbol}/status",
    response_model=TrainingJobResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Job não encontrado"}
    },
    summary="Status do treinamento",
    description="""
    Retorna o status atual do treinamento de um modelo.
    
    Inclui:
    - Status (queued, training, completed, failed)
    - Progresso (0-100%)
    - Época atual / Total de épocas
    - Métricas finais (se concluído)
    - Mensagem de erro (se falhou)
    """
)
async def get_training_status(
    symbol: str = Path(..., description="Símbolo da ação")
):
    """Retorna status do treinamento de um modelo."""
    symbol = symbol.upper()
    training_queue = get_training_queue()
    
    job = training_queue.get_job(symbol)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Nenhum job de treinamento encontrado para {symbol}"
        )
    
    return TrainingJobResponse(**job.to_dict())


# ==========================================
# Cancelar Treinamento
# ==========================================

@router.delete(
    "/{symbol}/training",
    responses={
        404: {"model": ErrorResponse, "description": "Job não encontrado"},
        400: {"model": ErrorResponse, "description": "Não pode cancelar"}
    },
    summary="Cancelar treinamento",
    description="""
    Cancela um treinamento na fila.
    
    **Nota:** Não é possível cancelar um treinamento já em execução.
    """
)
async def cancel_training(
    symbol: str = Path(..., description="Símbolo da ação")
):
    """Cancela treinamento na fila."""
    symbol = symbol.upper()
    training_queue = get_training_queue()
    
    job = training_queue.get_job(symbol)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Nenhum job encontrado para {symbol}"
        )
    
    if job.status == TrainingStatus.TRAINING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Não é possível cancelar treinamento em execução"
        )
    
    if job.status != TrainingStatus.QUEUED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job não está na fila. Status atual: {job.status.value}"
        )
    
    training_queue.cancel_job(symbol)
    
    return {"message": f"Treinamento de {symbol} cancelado", "symbol": symbol}


# ==========================================
# Remover Modelo
# ==========================================

@router.delete(
    "/{symbol}",
    response_model=ModelDeleteResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Modelo não encontrado"},
        400: {"model": ErrorResponse, "description": "Não pode remover"}
    },
    summary="Remover modelo",
    description="""
    Remove um modelo do sistema.
    
    **Ações:**
    - Descarrega modelo da memória
    - Remove arquivos do modelo
    - Remove do registro
    
    **Nota:** Não é possível remover modelo em treinamento.
    """
)
async def delete_model(
    symbol: str = Path(..., description="Símbolo da ação"),
    delete_files: bool = Query(True, description="Remover arquivos do modelo")
):
    """Remove modelo do sistema."""
    symbol = symbol.upper()
    registry = get_model_registry()
    training_queue = get_training_queue()
    settings = get_settings()
    
    # Verifica se está em treinamento
    job = training_queue.get_job(symbol)
    if job and job.status in [TrainingStatus.QUEUED, TrainingStatus.TRAINING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Não é possível remover {symbol} enquanto está em treinamento"
        )
    
    # Verifica se existe
    if not registry.has_model(symbol):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modelo não encontrado para {symbol}"
        )
    
    # Descarrega da memória e remove do registro
    registry.unregister_model(symbol)
    
    # Remove job da fila (se existir)
    training_queue.remove_job(symbol)
    
    # Remove arquivos
    files_deleted = False
    if delete_files:
        model_dir = os.path.join(settings.models_dir, symbol)
        if os.path.exists(model_dir):
            try:
                shutil.rmtree(model_dir)
                files_deleted = True
                logger.info(f"[{symbol}] Arquivos removidos: {model_dir}")
            except Exception as e:
                logger.error(f"[{symbol}] Erro ao remover arquivos: {e}")
    
    return ModelDeleteResponse(
        message=f"Modelo {symbol} removido com sucesso",
        symbol=symbol,
        files_deleted=files_deleted
    )


# ==========================================
# Recarregar Modelo
# ==========================================

@router.post(
    "/{symbol}/reload",
    responses={
        404: {"model": ErrorResponse, "description": "Modelo não encontrado"}
    },
    summary="Recarregar modelo",
    description="Recarrega um modelo na memória (útil após atualização de arquivos)."
)
async def reload_model(
    symbol: str = Path(..., description="Símbolo da ação")
):
    """Recarrega modelo na memória."""
    symbol = symbol.upper()
    registry = get_model_registry()
    
    # Descarrega se estiver carregado
    registry.unload_model(symbol)
    
    # Recarrega
    model = registry.get_model(symbol)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modelo não encontrado para {symbol}"
        )
    
    return {
        "message": f"Modelo {symbol} recarregado com sucesso",
        "symbol": symbol,
        "is_loaded": model.is_loaded
    }
