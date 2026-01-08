"""
Serviço de Predição
===================
Lógica de negócio para realizar previsões de preços.
Suporta modelo LSTM multivariado com indicadores técnicos.
Agora utiliza ModelRegistry para suporte a múltiplos modelos por símbolo.

FIAP - Tech Challenge Fase 4
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.core.model_registry import get_model_registry, ModelEntry
from app.config import get_settings
from model.data_collector import (
    add_technical_indicators, 
    FEATURE_COLUMNS, 
    get_latest_features,
    get_latest_features_async,
    StockDataCollector
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ThreadPoolExecutor para operações síncronas com yfinance
_executor = ThreadPoolExecutor(max_workers=2)


class ModelNotFoundError(Exception):
    """Exceção quando modelo não existe para o símbolo."""
    pass


class PredictionService:
    """
    Serviço para realizar previsões de preços de ações.
    
    Utiliza o ModelRegistry para obter modelos específicos por símbolo.
    Suporta modelo LSTM multivariado com indicadores técnicos.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.registry = get_model_registry()
        
    def _get_historical_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """
        Obtém dados históricos recentes de uma ação.
        Método SÍNCRONO - usar apenas em contextos que não conflitam com async.
        
        Args:
            symbol: Símbolo da ação
            days: Número de dias para buscar (com margem)
            
        Returns:
            DataFrame com dados históricos
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days * 1.5))
        
        try:
            collector = StockDataCollector(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            df = collector.download_data()
            
            if df.empty:
                raise ValueError(f"Nenhum dado encontrado para {symbol}")
                
            return df.tail(days)
            
        except Exception as e:
            logger.error(f"Erro ao buscar dados de {symbol}: {e}")
            raise
    
    def _get_model_for_symbol(self, symbol: str) -> ModelEntry:
        """
        Obtém modelo para um símbolo específico.
        
        Args:
            symbol: Símbolo da ação
            
        Returns:
            ModelEntry carregado
            
        Raises:
            ModelNotFoundError: Se modelo não existe para o símbolo
        """
        symbol = symbol.upper()
        model = self.registry.get_model(symbol)
        
        if model is None:
            # Verifica se está em treinamento
            from app.services.training_queue import get_training_queue, TrainingStatus
            queue = get_training_queue()
            job = queue.get_job(symbol)
            
            if job and job.status in [TrainingStatus.QUEUED, TrainingStatus.TRAINING]:
                raise ModelNotFoundError(
                    f"Modelo para {symbol} está em treinamento. "
                    f"Status: {job.status.value}, Progresso: {job.progress}%"
                )
            else:
                raise ModelNotFoundError(
                    f"Modelo não encontrado para {symbol}. "
                    f"Use POST /models/{symbol}/train para solicitar treinamento."
                )
        
        return model
    
    async def _get_historical_data_async(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """
        Versão assíncrona para obter dados históricos.
        Executa o download em thread separada para evitar problemas com curl_cffi.
        
        Args:
            symbol: Símbolo da ação
            days: Número de dias para buscar (com margem)
            
        Returns:
            DataFrame com dados históricos
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days * 1.5))
        
        try:
            collector = StockDataCollector(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            df = await collector.download_data_async()
            
            if df.empty:
                raise ValueError(f"Nenhum dado encontrado para {symbol}")
                
            return df.tail(days)
            
        except Exception as e:
            logger.error(f"Erro ao buscar dados de {symbol}: {e}")
            raise
    
    def _get_historical_data_with_features(self, symbol: str) -> Tuple[pd.DataFrame, float, datetime]:
        """
        Obtém dados históricos com indicadores técnicos.
        Método SÍNCRONO - usar apenas em contextos que não conflitam com async.
        
        Args:
            symbol: Símbolo da ação
            
        Returns:
            Tuple com (DataFrame de features, último preço, última data)
        """
        return get_latest_features(symbol, self.settings.sequence_length)
    
    async def _get_historical_data_with_features_async(self, symbol: str) -> Tuple[pd.DataFrame, float, datetime]:
        """
        Versão assíncrona para obter dados históricos com indicadores técnicos.
        Executa o download em thread separada para evitar problemas com curl_cffi.
        
        Args:
            symbol: Símbolo da ação
            
        Returns:
            Tuple com (DataFrame de features, último preço, última data)
        """
        return await get_latest_features_async(symbol, self.settings.sequence_length)
    
    def _get_business_days(self, start_date: datetime, num_days: int) -> List[str]:
        """
        Retorna lista de dias úteis futuros.
        
        Args:
            start_date: Data inicial
            num_days: Número de dias úteis
            
        Returns:
            Lista de datas no formato YYYY-MM-DD
        """
        dates = []
        current = start_date + timedelta(days=1)
        
        while len(dates) < num_days:
            # 0 = segunda, 6 = domingo
            if current.weekday() < 5:  # Dias úteis
                dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
            
        return dates
    
    def predict_from_symbol(
        self, 
        symbol: str, 
        days_ahead: int = 5
    ) -> Dict:
        """
        Realiza previsão buscando dados automaticamente.
        
        Args:
            symbol: Símbolo da ação
            days_ahead: Número de dias para prever
            
        Returns:
            Dicionário com previsões e metadados
            
        Raises:
            ModelNotFoundError: Se modelo não existe para o símbolo
        """
        symbol = symbol.upper()
        model = self._get_model_for_symbol(symbol)
        
        # Verifica se modelo é multivariado
        if model.is_multivariate():
            return self._predict_multivariate(symbol, days_ahead, model)
        else:
            return self._predict_univariate(symbol, days_ahead, model)
    
    async def predict_from_symbol_async(
        self, 
        symbol: str, 
        days_ahead: int = 5
    ) -> Dict:
        """
        Versão assíncrona de predict_from_symbol.
        Executa download de dados em thread separada para evitar problemas com curl_cffi.
        
        Args:
            symbol: Símbolo da ação
            days_ahead: Número de dias para prever
            
        Returns:
            Dicionário com previsões e metadados
            
        Raises:
            ModelNotFoundError: Se modelo não existe para o símbolo
        """
        symbol = symbol.upper()
        model = self._get_model_for_symbol(symbol)
        
        # Verifica se modelo é multivariado
        if model.is_multivariate():
            return await self._predict_multivariate_async(symbol, days_ahead, model)
        else:
            return await self._predict_univariate_async(symbol, days_ahead, model)
    
    def _predict_multivariate(self, symbol: str, days_ahead: int, model: ModelEntry) -> Dict:
        """
        Realiza previsão com modelo multivariado (indicadores técnicos).
        """
        # Busca dados com features
        features_df, last_close, last_date = self._get_historical_data_with_features(symbol)
        
        # Prepara sequência para predição
        sequence_length = self.settings.sequence_length
        feature_data = features_df.values  # (sequence_length, n_features)
        
        # Realiza previsões multi-step
        predictions = self._multi_step_predict_multivariate(feature_data, days_ahead, model)
        
        # Formata datas futuras
        future_dates = self._get_business_days(last_date.to_pydatetime(), days_ahead)
        
        # Monta resposta
        prediction_items = [
            {
                "date": future_dates[i],
                "day": i + 1,
                "predicted_close": round(predictions[i], 2)
            }
            for i in range(len(predictions))
        ]
        
        return {
            "symbol": symbol,
            "base_date": last_date.strftime('%Y-%m-%d'),
            "last_close": round(last_close, 2),
            "predictions": prediction_items,
            "model_version": self.settings.app_version,
            "model_type": "multivariate_lstm",
            "n_features": model.get_n_features(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _predict_univariate(self, symbol: str, days_ahead: int, model: ModelEntry) -> Dict:
        """
        Realiza previsão com modelo univariado (apenas preço).
        Mantém compatibilidade com modelo antigo.
        """
        # Busca dados históricos
        sequence_length = self.settings.sequence_length
        df = self._get_historical_data(symbol, days=sequence_length + 10)
        
        prices = df['Close'].values
        last_close = float(prices[-1])
        last_date = df.index[-1]
        
        # Prepara sequência para predição
        recent_prices = prices[-sequence_length:]
        
        # Realiza previsões multi-step
        predictions = self._multi_step_predict(recent_prices, days_ahead, model)
        
        # Formata datas futuras
        future_dates = self._get_business_days(last_date.to_pydatetime(), days_ahead)
        
        # Monta resposta
        prediction_items = [
            {
                "date": future_dates[i],
                "day": i + 1,
                "predicted_close": round(predictions[i], 2)
            }
            for i in range(len(predictions))
        ]
        
        return {
            "symbol": symbol,
            "base_date": last_date.strftime('%Y-%m-%d'),
            "last_close": round(last_close, 2),
            "predictions": prediction_items,
            "model_version": self.settings.app_version,
            "model_type": "univariate_lstm",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _predict_multivariate_async(self, symbol: str, days_ahead: int, model: ModelEntry) -> Dict:
        """
        Versão assíncrona de _predict_multivariate.
        Executa download em thread separada para evitar problemas com curl_cffi.
        """
        # Busca dados com features de forma assíncrona
        features_df, last_close, last_date = await self._get_historical_data_with_features_async(symbol)
        
        # Prepara sequência para predição
        sequence_length = self.settings.sequence_length
        feature_data = features_df.values  # (sequence_length, n_features)
        
        # Realiza previsões multi-step
        predictions = self._multi_step_predict_multivariate(feature_data, days_ahead, model)
        
        # Formata datas futuras
        future_dates = self._get_business_days(last_date.to_pydatetime(), days_ahead)
        
        # Monta resposta
        prediction_items = [
            {
                "date": future_dates[i],
                "day": i + 1,
                "predicted_close": round(predictions[i], 2)
            }
            for i in range(len(predictions))
        ]
        
        return {
            "symbol": symbol,
            "base_date": last_date.strftime('%Y-%m-%d'),
            "last_close": round(last_close, 2),
            "predictions": prediction_items,
            "model_version": self.settings.app_version,
            "model_type": "multivariate_lstm",
            "n_features": model.get_n_features(),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _predict_univariate_async(self, symbol: str, days_ahead: int, model: ModelEntry) -> Dict:
        """
        Versão assíncrona de _predict_univariate.
        Executa download em thread separada para evitar problemas com curl_cffi.
        """
        # Busca dados históricos de forma assíncrona
        sequence_length = self.settings.sequence_length
        df = await self._get_historical_data_async(symbol, days=sequence_length + 10)
        
        prices = df['Close'].values
        last_close = float(prices[-1])
        last_date = df.index[-1]
        
        # Prepara sequência para predição
        recent_prices = prices[-sequence_length:]
        
        # Realiza previsões multi-step
        predictions = self._multi_step_predict(recent_prices, days_ahead, model)
        
        # Formata datas futuras
        future_dates = self._get_business_days(last_date.to_pydatetime(), days_ahead)
        
        # Monta resposta
        prediction_items = [
            {
                "date": future_dates[i],
                "day": i + 1,
                "predicted_close": round(predictions[i], 2)
            }
            for i in range(len(predictions))
        ]
        
        return {
            "symbol": symbol,
            "base_date": last_date.strftime('%Y-%m-%d'),
            "last_close": round(last_close, 2),
            "predictions": prediction_items,
            "model_version": self.settings.app_version,
            "model_type": "univariate_lstm",
            "timestamp": datetime.now().isoformat()
        }
    
    def predict_from_prices(
        self,
        prices: List[float],
        days_ahead: int = 5
    ) -> Dict:
        """
        Realiza previsão com dados históricos fornecidos.
        
        Args:
            prices: Lista de preços históricos
            days_ahead: Número de dias para prever
            
        Returns:
            Dicionário com previsões
        """
        # Usa modelo padrão para predição com dados customizados
        default_symbol = self.settings.default_symbol
        model = self._get_model_for_symbol(default_symbol)
            
        sequence_length = self.settings.sequence_length
        
        if len(prices) < sequence_length:
            raise ValueError(
                f"Necessário pelo menos {sequence_length} preços históricos. "
                f"Recebido: {len(prices)}"
            )
        
        prices_array = np.array(prices)
        recent_prices = prices_array[-sequence_length:]
        
        # Realiza previsões
        predictions = self._multi_step_predict(recent_prices, days_ahead, model)
        
        # Gera datas futuras a partir de hoje
        future_dates = self._get_business_days(datetime.now(), days_ahead)
        
        prediction_items = [
            {
                "date": future_dates[i],
                "day": i + 1,
                "predicted_close": round(predictions[i], 2)
            }
            for i in range(len(predictions))
        ]
        
        return {
            "symbol": "CUSTOM",
            "base_date": datetime.now().strftime('%Y-%m-%d'),
            "last_close": round(float(prices[-1]), 2),
            "predictions": prediction_items,
            "model_version": self.settings.app_version,
            "timestamp": datetime.now().isoformat()
        }
    
    def _multi_step_predict(
        self, 
        initial_sequence: np.ndarray, 
        steps: int,
        model: ModelEntry
    ) -> List[float]:
        """
        Realiza predição multi-step recursiva (modelo univariado).
        
        Para prever N dias à frente, usa a predição do dia anterior
        como entrada para prever o próximo dia.
        
        Args:
            initial_sequence: Sequência inicial de preços
            steps: Número de passos para prever
            model: ModelEntry do modelo a usar
            
        Returns:
            Lista de previsões
        """
        predictions = []
        sequence = initial_sequence.copy()
        
        for _ in range(steps):
            # Normaliza sequência
            scaled = model.scale(sequence.reshape(-1, 1))
            
            # Reshape para LSTM: (1, sequence_length, 1)
            input_seq = scaled.reshape(1, len(scaled), 1)
            
            # Prediz próximo valor (normalizado)
            pred_scaled = model.predict(input_seq)
            
            # Desnormaliza
            pred = model.inverse_scale(pred_scaled)[0, 0]
            predictions.append(float(pred))
            
            # Atualiza sequência: remove primeiro, adiciona predição
            sequence = np.append(sequence[1:], pred)
        
        return predictions
    
    def _multi_step_predict_multivariate(
        self,
        initial_features: np.ndarray,
        steps: int,
        model: ModelEntry
    ) -> List[float]:
        """
        Realiza predição multi-step recursiva (modelo multivariado).
        
        Para prever N dias à frente, usa a predição do dia anterior
        como entrada para prever o próximo dia, atualizando apenas Close
        e mantendo os outros indicadores do último dia.
        
        Args:
            initial_features: Sequência inicial de features (sequence_length, n_features)
            steps: Número de passos para prever
            
        Returns:
            Lista de previsões
        """
        predictions = []
        features = initial_features.copy()
        
        for _ in range(steps):
            # Normaliza todas as features
            scaled = model.scale_features(features)
            
            # Reshape para LSTM: (1, sequence_length, n_features)
            input_seq = scaled.reshape(1, scaled.shape[0], scaled.shape[1])
            
            # Prediz próximo valor (normalizado)
            pred_scaled = model.predict(input_seq)
            
            # Desnormaliza (apenas Close)
            pred = model.inverse_scale(pred_scaled)[0, 0]
            predictions.append(float(pred))
            
            # Atualiza features: remove primeira linha, adiciona nova linha
            # Para simplificar, usamos os indicadores do último dia
            # atualizando apenas o preço de fechamento
            new_row = features[-1].copy()
            new_row[0] = pred  # Close é a primeira coluna
            
            # Atualiza indicadores baseados no novo preço (aproximação simples)
            # Log_Return
            if features[-1, 0] != 0:
                new_row[11] = np.log(pred / features[-1, 0])  # Log_Return
            
            # Momentum_10 (aproximação)
            if len(features) >= 10 and features[-10, 0] != 0:
                new_row[8] = pred / features[-10, 0] - 1
            
            # ROC (aproximação)
            if len(features) >= 10 and features[-10, 0] != 0:
                new_row[10] = (pred - features[-10, 0]) / features[-10, 0] * 100
            
            # Remove primeira linha e adiciona nova
            features = np.vstack([features[1:], new_row])
        
        return predictions


# Instância global do serviço
_prediction_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """
    Retorna instância do serviço de predição.
    
    Returns:
        PredictionService
    """
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service
