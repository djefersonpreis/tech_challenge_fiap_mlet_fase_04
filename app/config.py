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
    app_version: str = "2.0.0"
    app_description: str = "API para previsão de preços de ações usando LSTM Multivariado - FIAP Tech Challenge Fase 4"
    debug: bool = False
    
    # Modelo
    model_path: str = "models/lstm_model.keras"
    scaler_path: str = "models/scaler.pkl"
    scaler_features_path: str = "models/scaler_features.pkl"
    features_path: str = "models/feature_columns.json"
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
