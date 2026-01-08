"""
Configurações da Aplicação
==========================
Gerencia variáveis de ambiente e configurações.

FIAP - Tech Challenge Fase 4
"""

import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Configurações da aplicação."""
    
    # API
    app_name: str = "Stock Price Prediction API"
    app_version: str = "2.1.0"
    app_description: str = "API para previsão de preços de ações usando LSTM Multivariado com suporte a múltiplos modelos - FIAP Tech Challenge Fase 4"
    debug: bool = False
    
    # Diretório base de modelos
    models_dir: str = "models"
    
    # Modelo (mantido para compatibilidade, mas agora usa models_dir/{symbol}/)
    model_path: str = "models/PETR4.SA/lstm_model.keras"
    scaler_path: str = "models/PETR4.SA/scaler.pkl"
    scaler_features_path: str = "models/PETR4.SA/scaler_features.pkl"
    features_path: str = "models/PETR4.SA/feature_columns.json"
    sequence_length: int = 60
    default_symbol: str = "PETR4.SA"
    
    # Previsão
    max_prediction_days: int = 30
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Retorna instância cacheada das configurações.
    
    Returns:
        Objeto Settings com configurações
    """
    return Settings()
