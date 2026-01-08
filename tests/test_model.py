"""
Testes do Modelo
================
Testes unitários para os módulos do modelo LSTM.

FIAP - Tech Challenge Fase 4
"""

import pytest
import numpy as np
import sys
import os

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.preprocessor import DataPreprocessor
from model.evaluate import ModelEvaluator


class TestDataPreprocessor:
    """Testes para o preprocessador de dados."""
    
    def test_fit_transform(self):
        """Testa ajuste e transformação do scaler."""
        preprocessor = DataPreprocessor(sequence_length=10)
        data = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
        
        scaled = preprocessor.fit_transform(data)
        
        assert scaled.min() >= 0
        assert scaled.max() <= 1
        assert len(scaled) == len(data)
    
    def test_inverse_transform(self):
        """Testa reversão da normalização."""
        preprocessor = DataPreprocessor(sequence_length=10)
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0]).reshape(-1, 1)
        
        scaled = preprocessor.fit_transform(data)
        recovered = preprocessor.inverse_transform(scaled)
        
        np.testing.assert_array_almost_equal(data, recovered, decimal=5)
    
    def test_create_sequences(self):
        """Testa criação de sequências."""
        preprocessor = DataPreprocessor(sequence_length=3)
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
        
        preprocessor.fit(data)
        scaled = preprocessor.transform(data)
        X, y = preprocessor.create_sequences(scaled)
        
        assert X.shape[0] == 7  # 10 - 3 = 7 sequências
        assert X.shape[1] == 3  # sequence_length
        assert X.shape[2] == 1  # features
        assert len(y) == 7
    
    def test_prepare_train_test_split(self):
        """Testa split treino/teste."""
        preprocessor = DataPreprocessor(sequence_length=5)
        data = np.random.uniform(10, 50, 100)
        
        X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(
            data, train_ratio=0.8
        )
        
        assert len(X_train) > len(X_test)
        assert X_train.shape[1] == 5
        assert X_train.shape[2] == 1


class TestModelEvaluator:
    """Testes para o avaliador de modelos."""
    
    def test_mae(self):
        """Testa cálculo do MAE."""
        evaluator = ModelEvaluator()
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([11, 19, 31, 39, 51])
        
        mae = evaluator.mae(y_true, y_pred)
        
        assert mae == 1.0  # Média dos erros absolutos
    
    def test_rmse(self):
        """Testa cálculo do RMSE."""
        evaluator = ModelEvaluator()
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([10, 20, 30, 40, 50])  # Predição perfeita
        
        rmse = evaluator.rmse(y_true, y_pred)
        
        assert rmse == 0.0
    
    def test_mape(self):
        """Testa cálculo do MAPE."""
        evaluator = ModelEvaluator()
        y_true = np.array([100, 100, 100, 100])
        y_pred = np.array([90, 110, 95, 105])  # Erros de 10%, 10%, 5%, 5%
        
        mape = evaluator.mape(y_true, y_pred)
        
        assert abs(mape - 7.5) < 0.01  # 7.5% de erro médio
    
    def test_r2_perfect(self):
        """Testa R² com predição perfeita."""
        evaluator = ModelEvaluator()
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([10, 20, 30, 40, 50])
        
        r2 = evaluator.r2(y_true, y_pred)
        
        assert r2 == 1.0
    
    def test_calculate_all_metrics(self):
        """Testa cálculo de todas as métricas."""
        evaluator = ModelEvaluator()
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([11, 19, 31, 39, 51])
        
        metrics = evaluator.calculate_all_metrics(y_true, y_pred)
        
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert "r2" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
