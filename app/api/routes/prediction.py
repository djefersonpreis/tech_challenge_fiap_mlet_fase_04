"""
Rotas de Predição
=================
Endpoints para previsão de preços de ações.

FIAP - Tech Challenge Fase 4
"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime
import logging

from app.api.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    HistoricalDataRequest,
    ErrorResponse
)
from app.services.prediction_service import get_prediction_service
from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Predição"])


@router.post(
    "/",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Dados inválidos"},
        500: {"model": ErrorResponse, "description": "Erro interno"}
    },
    summary="Prever preços de ações",
    description="""
    Realiza previsão de preços de fechamento para uma ação.
    
    O endpoint busca automaticamente os dados históricos mais recentes
    da ação especificada e utiliza o modelo LSTM para prever os próximos
    N dias de fechamento.
    
    **Parâmetros:**
    - `symbol`: Símbolo da ação no Yahoo Finance (ex: PETR4.SA, VALE3.SA)
    - `days_ahead`: Número de dias úteis para prever (1-30)
    
    **Retorna:**
    - Lista de previsões com data e preço previsto
    - Último preço conhecido
    - Metadados da predição
    """
)
async def predict_stock_price(request: PredictionRequest):
    """
    Endpoint principal para previsão de preços.
    
    Args:
        request: PredictionRequest com symbol e days_ahead
        
    Returns:
        PredictionResponse com previsões
    """
    try:
        service = get_prediction_service()
        settings = get_settings()
        
        # Valida limite de dias
        if request.days_ahead > settings.max_prediction_days:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Máximo de {settings.max_prediction_days} dias permitido"
            )
        
        logger.info(f"Predição solicitada: {request.symbol} - {request.days_ahead} dias")
        
        # Usa versão assíncrona para evitar problemas com curl_cffi
        result = await service.predict_from_symbol_async(
            symbol=request.symbol,
            days_ahead=request.days_ahead
        )
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar predição: {str(e)}"
        )


@router.post(
    "/custom",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Dados inválidos"},
        500: {"model": ErrorResponse, "description": "Erro interno"}
    },
    summary="Prever com dados personalizados",
    description="""
    Realiza previsão usando dados históricos fornecidos pelo usuário.
    
    Útil quando você já possui os dados históricos ou quer testar
    com um conjunto de dados específico.
    
    **Parâmetros:**
    - `prices`: Lista com pelo menos 60 preços históricos
    - `days_ahead`: Número de dias para prever (1-30)
    
    **Nota:** Os preços devem estar em ordem cronológica (mais antigo primeiro).
    """
)
async def predict_from_custom_data(request: HistoricalDataRequest):
    """
    Endpoint para predição com dados fornecidos pelo usuário.
    
    Args:
        request: HistoricalDataRequest com prices e days_ahead
        
    Returns:
        PredictionResponse com previsões
    """
    try:
        service = get_prediction_service()
        settings = get_settings()
        
        # Valida quantidade mínima de dados
        if len(request.prices) < settings.sequence_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Necessário pelo menos {settings.sequence_length} preços históricos"
            )
        
        # Valida limite de dias
        if request.days_ahead > settings.max_prediction_days:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Máximo de {settings.max_prediction_days} dias permitido"
            )
        
        logger.info(f"Predição custom solicitada: {len(request.prices)} preços - {request.days_ahead} dias")
        
        result = service.predict_from_prices(
            prices=request.prices,
            days_ahead=request.days_ahead
        )
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Erro na predição custom: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar predição: {str(e)}"
        )
