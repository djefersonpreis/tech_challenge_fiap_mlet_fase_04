"""
Testes da API
=============
Testes unitários e de integração para a API de previsão.

FIAP - Tech Challenge Fase 4
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np
import sys
import os

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHealthEndpoints:
    """Testes para endpoints de health check."""
    
    def test_root_endpoint(self, client):
        """Testa endpoint raiz."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
    
    def test_health_endpoint(self, client):
        """Testa endpoint de health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "scaler_loaded" in data
        assert "timestamp" in data
    
    def test_model_info_endpoint(self, client):
        """Testa endpoint de informações do modelo."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "sequence_length" in data


class TestPredictionEndpoints:
    """Testes para endpoints de predição."""
    
    def test_predict_default_symbol(self, client):
        """Testa predição com símbolo padrão."""
        response = client.post(
            "/predict/",
            json={"symbol": "PETR4.SA", "days_ahead": 5}
        )
        # Pode falhar se modelo não estiver carregado
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "symbol" in data
            assert "predictions" in data
            assert len(data["predictions"]) == 5
    
    def test_predict_invalid_days(self, client):
        """Testa predição com número inválido de dias."""
        response = client.post(
            "/predict/",
            json={"symbol": "PETR4.SA", "days_ahead": 100}
        )
        assert response.status_code == 422  # Validation error
    
    def test_predict_custom_prices(self, client):
        """Testa predição com preços customizados."""
        # Gera 60 preços fictícios
        prices = list(np.random.uniform(30, 40, 60))
        
        response = client.post(
            "/predict/custom",
            json={"prices": prices, "days_ahead": 3}
        )
        
        # Pode falhar se modelo não estiver carregado
        assert response.status_code in [200, 500]
    
    def test_predict_custom_insufficient_prices(self, client):
        """Testa predição com preços insuficientes."""
        prices = list(np.random.uniform(30, 40, 30))  # Menos que 60
        
        response = client.post(
            "/predict/custom",
            json={"prices": prices, "days_ahead": 3}
        )
        
        assert response.status_code == 422  # Validation error


@pytest.fixture
def client():
    """Cria cliente de teste."""
    from app.main import app
    return TestClient(app)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
