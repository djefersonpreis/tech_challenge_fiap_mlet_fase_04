"""
Pydantic Schemas para Predição
==============================
Define os modelos de entrada e saída da API.

FIAP - Tech Challenge Fase 4
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime, date


class PredictionRequest(BaseModel):
    """
    Schema de requisição para predição.
    
    Attributes:
        symbol: Símbolo da ação (ex: PETR4.SA)
        days_ahead: Número de dias para prever (1-30)
    """
    symbol: str = Field(
        default="PETR4.SA",
        description="Símbolo da ação no Yahoo Finance",
        examples=["PETR4.SA", "VALE3.SA", "ITUB4.SA"]
    )
    days_ahead: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Número de dias para prever (máximo 30)"
    )
    
    @field_validator('symbol')
    @classmethod
    def symbol_must_be_uppercase(cls, v: str) -> str:
        return v.upper()


class PredictionItem(BaseModel):
    """Item individual de predição."""
    date: str = Field(description="Data da previsão (YYYY-MM-DD)")
    day: int = Field(description="Dia da previsão (1, 2, 3...)")
    predicted_close: float = Field(description="Preço de fechamento previsto")


class PredictionResponse(BaseModel):
    """
    Schema de resposta para predição.
    
    Attributes:
        symbol: Símbolo da ação
        predictions: Lista de previsões
        model_version: Versão do modelo
        timestamp: Timestamp da predição
    """
    symbol: str = Field(description="Símbolo da ação")
    base_date: str = Field(description="Data base para as previsões")
    last_close: float = Field(description="Último preço de fechamento conhecido")
    predictions: List[PredictionItem] = Field(description="Lista de previsões")
    model_version: str = Field(description="Versão do modelo utilizado")
    timestamp: str = Field(description="Timestamp da requisição")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "PETR4.SA",
                "base_date": "2026-01-03",
                "last_close": 38.50,
                "predictions": [
                    {"date": "2026-01-06", "day": 1, "predicted_close": 38.75},
                    {"date": "2026-01-07", "day": 2, "predicted_close": 38.90}
                ],
                "model_version": "1.0.0",
                "timestamp": "2026-01-04T10:30:00"
            }
        }
    }


class HistoricalDataRequest(BaseModel):
    """
    Schema para requisição com dados históricos fornecidos pelo usuário.
    """
    prices: List[float] = Field(
        min_length=60,
        description="Lista com preços históricos (mínimo 60 valores)"
    )
    days_ahead: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Número de dias para prever"
    )


class ModelInfoResponse(BaseModel):
    """Informações sobre o modelo."""
    model_name: str
    model_version: str
    symbol_trained: str
    sequence_length: int
    architecture: str
    last_training: Optional[str] = None
    metrics: Optional[dict] = None


class HealthResponse(BaseModel):
    """Resposta do health check."""
    status: str
    model_loaded: bool
    scaler_loaded: bool
    timestamp: str


class ErrorResponse(BaseModel):
    """Schema para erros."""
    error: str
    detail: str
    timestamp: str
