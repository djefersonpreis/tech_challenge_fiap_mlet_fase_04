"""
Módulo de Coleta de Dados
=========================
Responsável por coletar dados históricos de preços de ações utilizando yfinance.
Inclui geração de indicadores técnicos para o modelo LSTM multivariado.

FIAP - Tech Challenge Fase 4
Empresa: PETR4.SA (Petrobras)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests

# Configura yfinance para usar requests padrão em vez de curl_cffi
# Isso evita o erro "Impersonating chromeXXX is not supported"
_requests_session = requests.Session()
_requests_session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool para executar downloads de forma isolada
_executor = ThreadPoolExecutor(max_workers=4)

# Features utilizadas pelo modelo (deve corresponder ao treinamento)
FEATURE_COLUMNS = [
    'Close',           # Target principal
    'RSI',             # Momentum
    'MACD',            # Tendência
    'MACD_Hist',       # Força da tendência
    'BB_Position',     # Posição nas bandas
    'BB_Width',        # Largura das bandas (volatilidade)
    'ATR',             # Volatilidade
    'Volatility',      # Volatilidade histórica
    'Momentum_10',     # Momentum 10 dias
    'Volume_Norm',     # Volume relativo
    'ROC',             # Rate of Change
    'Log_Return'       # Retorno logarítmico
]


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona indicadores técnicos ao DataFrame.
    
    Args:
        df: DataFrame com dados OHLCV
        
    Returns:
        DataFrame com indicadores técnicos adicionados
    """
    df = df.copy()
    
    # RSI (Relative Strength Index) - 14 períodos
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # ATR (Average True Range) - Volatilidade
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # Retornos logarítmicos
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatilidade histórica (20 dias)
    df['Volatility'] = df['Log_Return'].rolling(window=20).std() * np.sqrt(252)
    
    # Momentum
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    
    # Volume normalizado
    df['Volume_Norm'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    # Price Rate of Change (ROC)
    df['ROC'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
    
    return df


class StockDataCollector:
    """
    Classe para coleta de dados históricos de ações.
    
    Attributes:
        symbol (str): Símbolo da ação (ex: 'PETR4.SA')
        start_date (str): Data inicial para coleta
        end_date (str): Data final para coleta
    """
    
    DEFAULT_SYMBOL = "PETR4.SA"
    
    def __init__(
        self,
        symbol: str = DEFAULT_SYMBOL,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Inicializa o coletor de dados.
        
        Args:
            symbol: Símbolo da ação no Yahoo Finance
            start_date: Data inicial no formato 'YYYY-MM-DD'
            end_date: Data final no formato 'YYYY-MM-DD'
        """
        self.symbol = symbol
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.start_date = start_date or (
            datetime.now() - timedelta(days=5*365)
        ).strftime('%Y-%m-%d')
    
    def _download_sync(self) -> pd.DataFrame:
        """
        Método interno síncrono para baixar dados.
        Usa sessão requests padrão para evitar problemas com curl_cffi.
        """
        # Usa yf.download com sessão requests customizada para evitar curl_cffi
        df = yf.download(
            str(self.symbol),
            start=self.start_date,
            end=self.end_date,
            progress=False,
            session=_requests_session  # Usa requests padrão em vez de curl_cffi
        )
        
        return df
        
    def download_data(self) -> pd.DataFrame:
        """
        Baixa dados históricos da ação.
        
        Returns:
            DataFrame com colunas: Open, High, Low, Close, Adj Close, Volume
            
        Raises:
            ValueError: Se não conseguir baixar os dados
        """
        logger.info(f"Baixando dados de {self.symbol} de {self.start_date} até {self.end_date}")
        
        try:
            # Executa o download em uma thread separada para evitar
            # problemas com curl_cffi no event loop do FastAPI
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._download_sync)
                df = future.result(timeout=60)  # Timeout de 60 segundos
            
            if df.empty:
                raise ValueError(f"Nenhum dado encontrado para {self.symbol}")
            
            # Remover MultiIndex se existir (yfinance às vezes retorna assim)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            logger.info(f"Dados baixados com sucesso: {len(df)} registros")
            return df
            
        except concurrent.futures.TimeoutError:
            logger.error(f"Timeout ao baixar dados de {self.symbol}")
            raise ValueError(f"Timeout ao baixar dados de {self.symbol}")
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Falha ao baixar dados de {self.symbol}: {e}")
    
    async def download_data_async(self) -> pd.DataFrame:
        """
        Versão assíncrona do download de dados.
        Executa o download em uma thread separada para não bloquear o event loop.
        
        Returns:
            DataFrame com colunas: Open, High, Low, Close, Volume
            
        Raises:
            ValueError: Se não conseguir baixar os dados
        """
        logger.info(f"[ASYNC] Baixando dados de {self.symbol} de {self.start_date} até {self.end_date}")
        
        try:
            # Executa em thread separada usando asyncio
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(_executor, self._download_sync)
            
            if df.empty:
                raise ValueError(f"Nenhum dado encontrado para {self.symbol}")
            
            # Remover MultiIndex se existir
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            logger.info(f"[ASYNC] Dados baixados com sucesso: {len(df)} registros")
            return df
            
        except Exception as e:
            logger.error(f"[ASYNC] Erro ao baixar dados: {e}")
            traceback.print_exc()
            raise ValueError(f"Falha ao baixar dados de {self.symbol}: {e}")
    
    def get_closing_prices(self) -> pd.Series:
        """
        Obtém apenas os preços de fechamento.
        
        Returns:
            Series com preços de fechamento indexados por data
        """
        df = self.download_data()
        return df['Close']
    
    def get_data_info(self) -> dict:
        """
        Retorna informações sobre os dados coletados.
        
        Returns:
            Dicionário com estatísticas dos dados
        """
        df = self.download_data()
        
        return {
            "symbol": self.symbol,
            "start_date": str(df.index.min().date()),
            "end_date": str(df.index.max().date()),
            "total_records": len(df),
            "columns": list(df.columns),
            "close_stats": {
                "min": float(df['Close'].min()),
                "max": float(df['Close'].max()),
                "mean": float(df['Close'].mean()),
                "std": float(df['Close'].std())
            }
        }


def get_latest_data(symbol: str = "PETR4.SA", days: int = 60) -> pd.DataFrame:
    """
    Função utilitária para obter os dados mais recentes de uma ação.
    
    Args:
        symbol: Símbolo da ação
        days: Número de dias para buscar
        
    Returns:
        DataFrame com dados históricos
    """
    end_date = datetime.now()
    # Adiciona margem para dias não-úteis
    start_date = end_date - timedelta(days=int(days * 1.5))
    
    collector = StockDataCollector(
        symbol=symbol,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    df = collector.download_data()
    return df.tail(days)


def get_latest_features(symbol: str = "PETR4.SA", sequence_length: int = 60) -> Tuple[pd.DataFrame, float]:
    """
    Obtém os dados mais recentes com indicadores técnicos para predição.
    
    Args:
        symbol: Símbolo da ação
        sequence_length: Número de timesteps necessários
        
    Returns:
        Tuple com (DataFrame com features, último preço de fechamento)
    """
    # Precisa de dados extras para calcular indicadores (30 dias de warmup)
    warmup_days = 50
    total_days = sequence_length + warmup_days
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(total_days * 1.5))
    
    collector = StockDataCollector(
        symbol=symbol,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # Baixa dados OHLCV
    df = collector.download_data()
    
    # Adiciona indicadores técnicos
    df = add_technical_indicators(df)
    
    # Remove NaN gerados pelos indicadores
    df_clean = df.dropna()
    
    # Retorna últimos sequence_length registros
    features_df = df_clean[FEATURE_COLUMNS].tail(sequence_length)
    last_close = float(df_clean['Close'].iloc[-1])
    last_date = df_clean.index[-1]
    
    logger.info(f"Features preparadas: {len(features_df)} registros, {len(FEATURE_COLUMNS)} features")
    
    return features_df, last_close, last_date


async def get_latest_data_async(symbol: str = "PETR4.SA", days: int = 60) -> pd.DataFrame:
    """
    Versão assíncrona para obter os dados mais recentes de uma ação.
    
    Args:
        symbol: Símbolo da ação
        days: Número de dias para buscar
        
    Returns:
        DataFrame com dados históricos
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(days * 1.5))
    
    collector = StockDataCollector(
        symbol=symbol,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    df = await collector.download_data_async()
    return df.tail(days)


async def get_latest_features_async(symbol: str = "PETR4.SA", sequence_length: int = 60) -> Tuple[pd.DataFrame, float, datetime]:
    """
    Versão assíncrona para obter dados com indicadores técnicos para predição.
    Executa o download em thread separada para evitar problemas com curl_cffi.
    
    Args:
        symbol: Símbolo da ação
        sequence_length: Número de timesteps necessários
        
    Returns:
        Tuple com (DataFrame com features, último preço de fechamento, última data)
    """
    warmup_days = 50
    total_days = sequence_length + warmup_days
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(total_days * 1.5))
    
    collector = StockDataCollector(
        symbol=symbol,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # Baixa dados OHLCV de forma assíncrona
    df = await collector.download_data_async()
    
    # Adiciona indicadores técnicos
    df = add_technical_indicators(df)
    
    # Remove NaN gerados pelos indicadores
    df_clean = df.dropna()
    
    # Retorna últimos sequence_length registros
    features_df = df_clean[FEATURE_COLUMNS].tail(sequence_length)
    last_close = float(df_clean['Close'].iloc[-1])
    last_date = df_clean.index[-1]
    
    logger.info(f"[ASYNC] Features preparadas: {len(features_df)} registros, {len(FEATURE_COLUMNS)} features")
    
    return features_df, last_close, last_date


if __name__ == "__main__":
    # Teste do módulo
    collector = StockDataCollector(symbol="PETR4.SA")
    info = collector.get_data_info()
    print("Informações dos dados:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Teste de features
    print("\nTestando geração de features...")
    features_df, last_close, last_date = get_latest_features("PETR4.SA", 60)
    print(f"Features shape: {features_df.shape}")
    print(f"Último preço: R$ {last_close:.2f}")
    print(f"Última data: {last_date}")
