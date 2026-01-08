"""
Módulo de Pré-processamento de Dados
====================================
Responsável pela normalização e preparação dos dados para o modelo LSTM.

FIAP - Tech Challenge Fase 4
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Classe para pré-processamento de dados de séries temporais.
    
    O LSTM requer dados normalizados (tipicamente entre 0 e 1) para
    convergência adequada durante o treinamento.
    
    Attributes:
        sequence_length (int): Número de timesteps para cada sequência
        scaler (MinMaxScaler): Normalizador dos dados
    """
    
    DEFAULT_SEQUENCE_LENGTH = 60  # 60 dias de histórico
    
    def __init__(self, sequence_length: int = DEFAULT_SEQUENCE_LENGTH):
        """
        Inicializa o preprocessador.
        
        Args:
            sequence_length: Número de timesteps para criar sequências
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._is_fitted = False
        
    def fit(self, data: np.ndarray) -> 'DataPreprocessor':
        """
        Ajusta o scaler aos dados.
        
        Args:
            data: Array numpy com os dados de preço
            
        Returns:
            Self para permitir method chaining
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        self.scaler.fit(data)
        self._is_fitted = True
        logger.info(f"Scaler ajustado - Min: {self.scaler.data_min_[0]:.2f}, Max: {self.scaler.data_max_[0]:.2f}")
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transforma os dados usando o scaler ajustado.
        
        Args:
            data: Array numpy com os dados de preço
            
        Returns:
            Dados normalizados entre 0 e 1
        """
        if not self._is_fitted:
            raise ValueError("Scaler não foi ajustado. Chame fit() primeiro.")
            
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        return self.scaler.transform(data)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Ajusta o scaler e transforma os dados em uma única operação.
        
        Args:
            data: Array numpy com os dados de preço
            
        Returns:
            Dados normalizados entre 0 e 1
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Reverte a normalização para obter valores originais.
        
        Args:
            data: Dados normalizados
            
        Returns:
            Dados na escala original
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        return self.scaler.inverse_transform(data)
    
    def create_sequences(
        self, 
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria sequências para treinamento do LSTM.
        
        Para cada ponto i, cria uma sequência X com os últimos
        sequence_length valores e y como o próximo valor.
        
        Args:
            data: Dados normalizados
            
        Returns:
            Tuple (X, y) onde:
                X: shape (samples, sequence_length, 1)
                y: shape (samples, 1)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i, 0])
            y.append(data[i, 0])
            
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X para formato LSTM: (samples, timesteps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        logger.info(f"Sequências criadas - X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    def prepare_train_test_split(
        self,
        data: np.ndarray,
        train_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepara dados para treinamento e teste.
        
        IMPORTANTE: Em séries temporais, NÃO podemos embaralhar os dados.
        O split deve ser cronológico.
        
        Args:
            data: Dados brutos de preço
            train_ratio: Proporção para treinamento (default: 80%)
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        # Calcula índice de divisão
        train_size = int(len(data) * train_ratio)
        
        # Divide ANTES de normalizar para evitar data leakage
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        logger.info(f"Split dos dados - Treino: {len(train_data)}, Teste: {len(test_data)}")
        
        # Ajusta scaler apenas nos dados de treino
        self.fit(train_data)
        
        # Normaliza os dados
        train_scaled = self.transform(train_data)
        test_scaled = self.transform(test_data)
        
        # Cria sequências
        X_train, y_train = self.create_sequences(train_scaled)
        
        # Para o teste, precisamos incluir parte do treino para formar as primeiras sequências
        full_test = np.concatenate([train_scaled[-self.sequence_length:], test_scaled])
        X_test, y_test = self.create_sequences(full_test)
        
        return X_train, X_test, y_train, y_test
    
    def prepare_prediction_sequence(self, data: np.ndarray) -> np.ndarray:
        """
        Prepara uma sequência para predição.
        
        Args:
            data: Últimos N valores de preço (N >= sequence_length)
            
        Returns:
            Sequência formatada para predição: shape (1, sequence_length, 1)
        """
        if len(data) < self.sequence_length:
            raise ValueError(
                f"Dados insuficientes. Necessário: {self.sequence_length}, Recebido: {len(data)}"
            )
        
        # Pega apenas os últimos sequence_length valores
        recent_data = data[-self.sequence_length:]
        
        if recent_data.ndim == 1:
            recent_data = recent_data.reshape(-1, 1)
        
        # Normaliza
        scaled_data = self.transform(recent_data)
        
        # Reshape para formato LSTM
        sequence = scaled_data.reshape(1, self.sequence_length, 1)
        
        return sequence
    
    def save_scaler(self, filepath: str) -> None:
        """
        Salva o scaler ajustado em arquivo.
        
        Args:
            filepath: Caminho para salvar o arquivo .pkl
        """
        if not self._is_fitted:
            raise ValueError("Scaler não foi ajustado. Nada para salvar.")
            
        joblib.dump(self.scaler, filepath)
        logger.info(f"Scaler salvo em: {filepath}")
    
    def load_scaler(self, filepath: str) -> None:
        """
        Carrega um scaler previamente salvo.
        
        Args:
            filepath: Caminho do arquivo .pkl
        """
        self.scaler = joblib.load(filepath)
        self._is_fitted = True
        logger.info(f"Scaler carregado de: {filepath}")


if __name__ == "__main__":
    # Teste do módulo
    from data_collector import StockDataCollector
    
    # Coleta dados
    collector = StockDataCollector(symbol="PETR4.SA")
    df = collector.download_data()
    prices = df['Close'].values
    
    # Preprocessa
    preprocessor = DataPreprocessor(sequence_length=60)
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(prices)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
