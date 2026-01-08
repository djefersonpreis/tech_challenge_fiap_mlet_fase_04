"""
Carregador de Modelo
====================
Gerencia o carregamento e cache do modelo LSTM multivariado.
Agora utiliza o ModelRegistry para suporte a múltiplos modelos.

FIAP - Tech Challenge Fase 4
"""

import os
import json
import logging
from functools import lru_cache
from typing import Optional, Tuple, List

import numpy as np
from tensorflow.keras.models import load_model
import joblib

from app.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduz logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ModelLoader:
    """
    Wrapper de compatibilidade que usa ModelRegistry internamente.
    
    Mantém a interface antiga para não quebrar código existente,
    mas delega para o ModelRegistry quando possível.
    
    Attributes:
        model: Modelo Keras carregado
        scaler_features: MinMaxScaler para features
        scaler_target: MinMaxScaler para target (Close)
        feature_columns: Lista de features usadas no treinamento
        is_loaded: Flag indicando se o modelo está carregado
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.model = None
        self.scaler_features = None
        self.scaler_target = None
        self.feature_columns = None
        self.is_loaded = False
        self.settings = get_settings()
        self._current_symbol = None
        self._initialized = True
        
    def load(self, symbol: str = None) -> bool:
        """
        Carrega o modelo e scalers dos arquivos.
        
        Args:
            symbol: Símbolo da ação (se None, usa símbolo padrão)
        
        Returns:
            True se carregado com sucesso
        """
        if symbol is None:
            symbol = self.settings.default_symbol
        
        symbol = symbol.upper()
        
        # Se já carregou este símbolo, retorna
        if self.is_loaded and self._current_symbol == symbol:
            return True
        
        try:
            # Determina diretório do modelo
            model_dir = os.path.join(self.settings.models_dir, symbol)
            
            # Fallback para estrutura antiga (arquivos na raiz de models/)
            if not os.path.exists(model_dir):
                model_dir = self.settings.models_dir
                logger.warning(f"Usando estrutura antiga de modelos em: {model_dir}")
            
            # Carrega modelo
            model_path = os.path.join(model_dir, "lstm_model.keras")
            if not os.path.exists(model_path):
                logger.error(f"Arquivo do modelo não encontrado: {model_path}")
                return False
                
            logger.info(f"Carregando modelo de: {model_path}")
            self.model = load_model(model_path)
            logger.info("Modelo carregado com sucesso!")
            
            # Carrega scaler de features
            scaler_features_path = os.path.join(model_dir, "scaler_features.pkl")
            if os.path.exists(scaler_features_path):
                logger.info(f"Carregando scaler de features de: {scaler_features_path}")
                self.scaler_features = joblib.load(scaler_features_path)
                logger.info("Scaler de features carregado!")
            else:
                self.scaler_features = None
            
            # Carrega scaler do target
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            if not os.path.exists(scaler_path):
                logger.error(f"Arquivo do scaler não encontrado: {scaler_path}")
                return False
                
            logger.info(f"Carregando scaler de target de: {scaler_path}")
            self.scaler_target = joblib.load(scaler_path)
            logger.info("Scaler de target carregado!")
            
            # Carrega lista de features
            features_path = os.path.join(model_dir, "feature_columns.json")
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.feature_columns = json.load(f)
                logger.info(f"Features carregadas: {len(self.feature_columns)} colunas")
            else:
                # Fallback para features padrão
                self.feature_columns = [
                    'Close', 'RSI', 'MACD', 'MACD_Hist', 'BB_Position', 'BB_Width',
                    'ATR', 'Volatility', 'Momentum_10', 'Volume_Norm', 'ROC', 'Log_Return'
                ]
                logger.warning(f"Usando features padrão: {self.feature_columns}")
            
            self.is_loaded = True
            self._current_symbol = symbol
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            self.is_loaded = False
            return False
    
    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """
        Realiza predição com o modelo.
        
        Args:
            sequence: Sequência normalizada - shape (1, sequence_length, n_features)
            
        Returns:
            Predição normalizada
        """
        if not self.is_loaded:
            raise RuntimeError("Modelo não carregado. Chame load() primeiro.")
            
        return self.model.predict(sequence, verbose=0)
    
    def scale_features(self, data: np.ndarray) -> np.ndarray:
        """
        Normaliza dados de features usando o scaler de features.
        
        Args:
            data: Dados brutos (n_samples, n_features)
            
        Returns:
            Dados normalizados
        """
        if not self.is_loaded:
            raise RuntimeError("Scaler não carregado. Chame load() primeiro.")
        
        if self.scaler_features is not None:
            return self.scaler_features.transform(data)
        else:
            # Fallback: normaliza apenas Close (compatibilidade com modelo antigo)
            return self.scaler_target.transform(data[:, 0].reshape(-1, 1))
    
    def scale(self, data: np.ndarray) -> np.ndarray:
        """
        Normaliza dados usando o scaler do target (compatibilidade).
        
        Args:
            data: Dados brutos
            
        Returns:
            Dados normalizados
        """
        if not self.is_loaded:
            raise RuntimeError("Scaler não carregado. Chame load() primeiro.")
            
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return self.scaler_target.transform(data)
    
    def inverse_scale(self, data: np.ndarray) -> np.ndarray:
        """
        Desnormaliza dados do target.
        
        Args:
            data: Dados normalizados
            
        Returns:
            Dados na escala original
        """
        if not self.is_loaded:
            raise RuntimeError("Scaler não carregado. Chame load() primeiro.")
            
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return self.scaler_target.inverse_transform(data)
    
    def get_n_features(self) -> int:
        """Retorna número de features do modelo."""
        if self.feature_columns:
            return len(self.feature_columns)
        return 1  # Fallback para modelo univariado
    
    def is_multivariate(self) -> bool:
        """Verifica se o modelo é multivariado."""
        return self.scaler_features is not None and self.get_n_features() > 1
    
    def get_status(self) -> dict:
        """
        Retorna status do carregamento.
        
        Returns:
            Dicionário com status
        """
        return {
            "model_loaded": self.model is not None,
            "scaler_features_loaded": self.scaler_features is not None,
            "scaler_target_loaded": self.scaler_target is not None,
            "is_ready": self.is_loaded,
            "is_multivariate": self.is_multivariate(),
            "n_features": self.get_n_features(),
            "feature_columns": self.feature_columns,
            "current_symbol": self._current_symbol
        }


@lru_cache()
def get_model_loader() -> ModelLoader:
    """
    Retorna instância cacheada do ModelLoader.
    
    Returns:
        ModelLoader singleton
    """
    loader = ModelLoader()
    return loader


def init_model(symbol: str = None) -> ModelLoader:
    """
    Inicializa e carrega o modelo.
    
    Args:
        symbol: Símbolo da ação (se None, usa padrão)
    
    Returns:
        ModelLoader com modelo carregado
    """
    loader = get_model_loader()
    if not loader.is_loaded or (symbol and symbol.upper() != loader._current_symbol):
        loader.load(symbol)
    return loader
