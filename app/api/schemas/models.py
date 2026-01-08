"""
Pydantic Schemas para Gerenciamento de Modelos
==============================================
Define os modelos de entrada e saída para o CRUD de modelos.

FIAP - Tech Challenge Fase 4
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ModelStatus(str, Enum):
    """Status possíveis de um modelo."""
    READY = "ready"
    TRAINING = "training"
    QUEUED = "queued"
    FAILED = "failed"


class TrainingStatus(str, Enum):
    """Status de treinamento."""
    QUEUED = "queued"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ==========================================
# Request Schemas
# ==========================================

class TrainingRequest(BaseModel):
    """
    Requisição para treinar um novo modelo.
    
    Attributes:
        epochs: Número máximo de épocas de treinamento
        start_date: Data inicial para coleta de dados históricos
        batch_size: Tamanho do batch para treinamento
    """
    epochs: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Número máximo de épocas (10-500)"
    )
    start_date: str = Field(
        default="2018-01-01",
        description="Data inicial para dados históricos (YYYY-MM-DD)",
        examples=["2018-01-01", "2020-01-01"]
    )
    batch_size: int = Field(
        default=32,
        ge=8,
        le=128,
        description="Tamanho do batch (8-128)"
    )
    
    @field_validator('start_date')
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Data deve estar no formato YYYY-MM-DD")
        return v


# ==========================================
# Response Schemas
# ==========================================

class ModelMetrics(BaseModel):
    """Métricas de avaliação do modelo."""
    mae: Optional[float] = Field(None, description="Mean Absolute Error")
    rmse: Optional[float] = Field(None, description="Root Mean Square Error")
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error (%)")
    r2: Optional[float] = Field(None, description="R² Score")


class ModelParameters(BaseModel):
    """Parâmetros de treinamento do modelo."""
    sequence_length: Optional[int] = Field(None, description="Tamanho da sequência LSTM")
    epochs_configured: Optional[int] = Field(None, description="Épocas configuradas")
    epochs_run: Optional[int] = Field(None, description="Épocas executadas")
    batch_size: Optional[int] = Field(None, description="Tamanho do batch")
    architecture: Optional[str] = Field(None, description="Arquitetura do modelo")
    lstm_units: Optional[List[int]] = Field(None, description="Unidades LSTM por camada")
    dropout_rate: Optional[float] = Field(None, description="Taxa de dropout")
    learning_rate: Optional[float] = Field(None, description="Taxa de aprendizado")


class ModelDataInfo(BaseModel):
    """Informações sobre os dados de treinamento."""
    total_records: Optional[int] = Field(None, description="Total de registros")
    train_records: Optional[int] = Field(None, description="Registros de treino")
    test_records: Optional[int] = Field(None, description="Registros de teste")
    price_min: Optional[float] = Field(None, description="Preço mínimo")
    price_max: Optional[float] = Field(None, description="Preço máximo")
    price_mean: Optional[float] = Field(None, description="Preço médio")


class ModelSummary(BaseModel):
    """Resumo de um modelo (para listagem)."""
    symbol: str = Field(description="Símbolo da ação")
    status: str = Field(description="Status do modelo (ready, training, queued, failed)")
    trained_at: Optional[str] = Field(None, description="Data/hora do treinamento")
    metrics: Optional[ModelMetrics] = Field(None, description="Métricas de avaliação")
    is_loaded_in_memory: bool = Field(default=False, description="Se está carregado na memória")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "PETR4.SA",
                "status": "ready",
                "trained_at": "2026-01-05T15:41:32",
                "metrics": {
                    "mae": 0.828,
                    "rmse": 1.052,
                    "mape": 2.65,
                    "r2": 0.548
                },
                "is_loaded_in_memory": True
            }
        }
    }


class ModelListResponse(BaseModel):
    """Resposta com lista de modelos."""
    models: List[ModelSummary] = Field(description="Lista de modelos")
    total: int = Field(description="Total de modelos")
    ready_count: int = Field(description="Modelos prontos para uso")
    training_count: int = Field(description="Modelos em treinamento/fila")


class ModelDetailResponse(BaseModel):
    """Detalhes completos de um modelo."""
    symbol: str = Field(description="Símbolo da ação")
    status: str = Field(description="Status do modelo")
    trained_at: Optional[str] = Field(None, description="Data/hora do treinamento")
    is_loaded_in_memory: bool = Field(default=False, description="Se está carregado")
    
    # Métricas
    metrics: Optional[ModelMetrics] = Field(None, description="Métricas de avaliação")
    
    # Parâmetros
    parameters: Optional[ModelParameters] = Field(None, description="Parâmetros de treinamento")
    
    # Dados
    data_info: Optional[ModelDataInfo] = Field(None, description="Info dos dados")
    
    # Features
    features: Optional[List[str]] = Field(None, description="Lista de features utilizadas")
    n_features: Optional[int] = Field(None, description="Número de features")
    
    # Histórico de treinamento
    training_history: Optional[Dict[str, Any]] = Field(None, description="Histórico do treinamento")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "PETR4.SA",
                "status": "ready",
                "trained_at": "2026-01-05T15:41:32",
                "is_loaded_in_memory": True,
                "metrics": {
                    "mae": 0.828,
                    "rmse": 1.052,
                    "mape": 2.65,
                    "r2": 0.548
                },
                "parameters": {
                    "sequence_length": 60,
                    "epochs_configured": 150,
                    "epochs_run": 89,
                    "batch_size": 32,
                    "architecture": "Bidirectional LSTM + BatchNorm"
                },
                "features": ["Close", "RSI", "MACD"],
                "n_features": 12
            }
        }
    }


class TrainingJobResponse(BaseModel):
    """Resposta com status de um job de treinamento."""
    symbol: str = Field(description="Símbolo da ação")
    status: str = Field(description="Status do treinamento")
    created_at: str = Field(description="Data/hora de criação")
    started_at: Optional[str] = Field(None, description="Data/hora de início")
    completed_at: Optional[str] = Field(None, description="Data/hora de conclusão")
    progress: int = Field(default=0, description="Progresso (0-100%)")
    epochs_total: int = Field(description="Total de épocas configuradas")
    epochs_current: int = Field(default=0, description="Época atual")
    error_message: Optional[str] = Field(None, description="Mensagem de erro (se falhou)")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parâmetros do treinamento")
    result_metrics: Optional[ModelMetrics] = Field(None, description="Métricas (se concluído)")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "VALE3.SA",
                "status": "training",
                "created_at": "2026-01-08T10:00:00",
                "started_at": "2026-01-08T10:00:05",
                "progress": 45,
                "epochs_total": 100,
                "epochs_current": 45,
                "parameters": {
                    "epochs": 100,
                    "start_date": "2018-01-01",
                    "batch_size": 32
                }
            }
        }
    }


class TrainingQueueResponse(BaseModel):
    """Resposta com status da fila de treinamento."""
    worker_running: bool = Field(description="Se o worker está ativo")
    current_job: Optional[str] = Field(None, description="Símbolo em treinamento")
    queued_count: int = Field(description="Quantidade na fila")
    training_count: int = Field(description="Em treinamento (0 ou 1)")
    completed_count: int = Field(description="Concluídos")
    failed_count: int = Field(description="Falhados")
    queued_symbols: List[str] = Field(description="Símbolos na fila")
    jobs: List[TrainingJobResponse] = Field(description="Lista de jobs")


class TrainingEnqueueResponse(BaseModel):
    """Resposta ao enfileirar treinamento."""
    message: str = Field(description="Mensagem de status")
    job: TrainingJobResponse = Field(description="Dados do job")
    queue_position: int = Field(description="Posição na fila")


class ModelDeleteResponse(BaseModel):
    """Resposta ao deletar modelo."""
    message: str = Field(description="Mensagem de status")
    symbol: str = Field(description="Símbolo removido")
    files_deleted: bool = Field(description="Se os arquivos foram removidos")


class ErrorResponse(BaseModel):
    """Schema para erros."""
    error: str = Field(description="Tipo do erro")
    detail: str = Field(description="Detalhes do erro")
    timestamp: str = Field(description="Timestamp do erro")
