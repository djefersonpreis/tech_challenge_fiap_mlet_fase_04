"""
Rotas de Health Check
=====================
Endpoints para verificação de saúde da API.

FIAP - Tech Challenge Fase 4
"""

from fastapi import APIRouter
from datetime import datetime
import logging
import json
import os

from app.api.schemas.prediction import HealthResponse, ModelInfoResponse
from app.core.model_loader import get_model_loader
from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health & Info"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificar saúde da API",
    description="Verifica se a API está funcionando e se o modelo está carregado."
)
async def health_check():
    """
    Endpoint de health check.
    
    Returns:
        HealthResponse com status dos componentes
    """
    loader = get_model_loader()
    status_info = loader.get_status()
    
    return HealthResponse(
        status="healthy" if status_info["is_ready"] else "degraded",
        model_loaded=status_info["model_loaded"],
        scaler_features_loaded=status_info["scaler_features_loaded"],
        scaler_target_loaded=status_info["scaler_target_loaded"],
        timestamp=datetime.now().isoformat()
    )


@router.get(
    "/",
    summary="Raiz da API",
    description="Endpoint raiz com informações básicas da API."
)
async def root():
    """
    Endpoint raiz com boas-vindas.
    
    Returns:
        Mensagem de boas-vindas e links úteis
    """
    settings = get_settings()
    
    return {
        "message": f"Bem-vindo à {settings.app_name}",
        "version": settings.app_version,
        "description": settings.app_description,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "metrics": "/metrics"
    }


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Informações do modelo",
    description="Retorna informações detalhadas sobre o modelo LSTM carregado."
)
async def model_info():
    """
    Retorna informações sobre o modelo.
    
    Returns:
        ModelInfoResponse com detalhes do modelo
    """
    settings = get_settings()
    loader = get_model_loader()
    
    # Tenta carregar métricas do treinamento
    metrics = None
    training_date = None
    
    try:
        results_path = os.path.join(
            os.path.dirname(settings.model_path), 
            "training_results.json"
        )
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
                metrics = results.get("evaluation_metrics")
                training_date = results.get("training_end")
    except Exception as e:
        logger.warning(f"Não foi possível carregar métricas: {e}")
    
    # Arquitetura do modelo
    architecture = "LSTM(128) → Dropout → LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(25) → Dense(1)"
    
    return ModelInfoResponse(
        model_name="LSTM Stock Price Predictor",
        model_version=settings.app_version,
        symbol_trained=settings.default_symbol,
        sequence_length=settings.sequence_length,
        architecture=architecture,
        last_training=training_date,
        metrics=metrics
    )
