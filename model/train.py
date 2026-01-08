"""
Script de Treinamento do Modelo LSTM
====================================
Executa o pipeline completo de treinamento do modelo.

FIAP - Tech Challenge Fase 4

Uso:
    python train.py [--symbol SYMBOL] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
"""

import argparse
import os
import sys
from datetime import datetime
import numpy as np
import json
import logging

# Adiciona o diretório pai ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.data_collector import StockDataCollector
from model.preprocessor import DataPreprocessor
from model.lstm_model import LSTMStockPredictor
from model.evaluate import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(
    symbol: str = "PETR4.SA",
    start_date: str = "2018-01-01",
    end_date: str = None,
    sequence_length: int = 60,
    epochs: int = 100,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    model_dir: str = "models",
    progress_callback = None
) -> dict:
    """
    Executa o pipeline completo de treinamento.
    
    Args:
        symbol: Símbolo da ação
        start_date: Data inicial para coleta de dados
        end_date: Data final para coleta de dados
        sequence_length: Tamanho da sequência LSTM
        epochs: Número máximo de épocas
        batch_size: Tamanho do batch
        train_ratio: Proporção de dados para treino
        model_dir: Diretório base para salvar o modelo (será criado subdiretório com nome do symbol)
        progress_callback: Função callback(epoch, total) para reportar progresso
        
    Returns:
        Dicionário com métricas e informações do treinamento
    """
    # Normaliza símbolo
    symbol = symbol.upper()
    
    # Cria diretório específico para o símbolo
    symbol_dir = os.path.join(model_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    logger.info(f"Diretório de saída: {symbol_dir}")
    results = {
        "symbol": symbol,
        "training_start": datetime.now().isoformat(),
        "parameters": {
            "sequence_length": sequence_length,
            "epochs": epochs,
            "batch_size": batch_size,
            "train_ratio": train_ratio
        }
    }
    
    # =========================================
    # 1. COLETA DE DADOS
    # =========================================
    logger.info("=" * 50)
    logger.info("ETAPA 1: Coleta de Dados")
    logger.info("=" * 50)
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    collector = StockDataCollector(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    df = collector.download_data()
    prices = df['Close'].values
    
    results["data_info"] = {
        "total_records": len(df),
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "price_min": float(prices.min()),
        "price_max": float(prices.max()),
        "price_mean": float(prices.mean())
    }
    
    logger.info(f"Dados coletados: {len(df)} registros")
    logger.info(f"Período: {df.index.min().date()} até {df.index.max().date()}")
    
    # =========================================
    # 2. PRÉ-PROCESSAMENTO
    # =========================================
    logger.info("=" * 50)
    logger.info("ETAPA 2: Pré-processamento")
    logger.info("=" * 50)
    
    preprocessor = DataPreprocessor(sequence_length=sequence_length)
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(
        prices, 
        train_ratio=train_ratio
    )
    
    results["data_split"] = {
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "y_train_shape": list(y_train.shape),
        "y_test_shape": list(y_test.shape)
    }
    
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    
    # =========================================
    # 3. CONSTRUÇÃO DO MODELO
    # =========================================
    logger.info("=" * 50)
    logger.info("ETAPA 3: Construção do Modelo")
    logger.info("=" * 50)
    
    model = LSTMStockPredictor(
        sequence_length=sequence_length,
        n_features=1,
        lstm_units=[128, 64, 32],
        dropout_rate=0.2,
        dense_units=25,
        learning_rate=0.001
    )
    model.build_model()
    
    results["model_info"] = model.get_model_summary()
    
    # =========================================
    # 4. TREINAMENTO
    # =========================================
    logger.info("=" * 50)
    logger.info("ETAPA 4: Treinamento")
    logger.info("=" * 50)
    
    # Usa diretório específico do símbolo
    checkpoint_path = os.path.join(symbol_dir, "checkpoint.keras")
    
    # Callback customizado para reportar progresso
    class ProgressCallback:
        def __init__(self, callback, total_epochs):
            self.callback = callback
            self.total_epochs = total_epochs
        
        def on_epoch_end(self, epoch, logs=None):
            if self.callback:
                self.callback(epoch + 1, self.total_epochs)
    
    progress_cb = ProgressCallback(progress_callback, epochs) if progress_callback else None
    
    history = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        progress_callback=progress_cb
    )
    
    results["training_history"] = {
        "epochs_run": len(history.history['loss']),
        "final_loss": float(history.history['loss'][-1]),
        "final_val_loss": float(history.history['val_loss'][-1]),
        "final_mae": float(history.history['mae'][-1]),
        "final_val_mae": float(history.history['val_mae'][-1]),
        "best_val_loss": float(min(history.history['val_loss']))
    }
    
    # =========================================
    # 5. AVALIAÇÃO
    # =========================================
    logger.info("=" * 50)
    logger.info("ETAPA 5: Avaliação do Modelo")
    logger.info("=" * 50)
    
    # Predições no conjunto de teste
    y_pred_scaled = model.predict(X_test)
    
    # Reverte normalização
    y_pred = preprocessor.inverse_transform(y_pred_scaled).flatten()
    y_true = preprocessor.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calcula métricas
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_all_metrics(y_true, y_pred)
    
    results["evaluation_metrics"] = metrics
    
    logger.info("Métricas de Avaliação:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # =========================================
    # 6. SALVAMENTO
    # =========================================
    logger.info("=" * 50)
    logger.info("ETAPA 6: Salvamento do Modelo")
    logger.info("=" * 50)
    
    # Salva modelo final no diretório do símbolo
    model_path = os.path.join(symbol_dir, "lstm_model.keras")
    model.save_model(model_path)
    
    # Salva scaler
    scaler_path = os.path.join(symbol_dir, "scaler.pkl")
    preprocessor.save_scaler(scaler_path)
    
    # Salva resultados do treinamento
    results["training_end"] = datetime.now().isoformat()
    results["saved_files"] = {
        "model": model_path,
        "scaler": scaler_path
    }
    
    # Função auxiliar para converter numpy types para JSON
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def convert_dict(d):
        if isinstance(d, dict):
            return {k: convert_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_dict(item) for item in d]
        else:
            return convert_numpy(d)
    
    results_serializable = convert_dict(results)
    
    results_path = os.path.join(symbol_dir, "training_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Modelo salvo em: {model_path}")
    logger.info(f"Scaler salvo em: {scaler_path}")
    logger.info(f"Resultados salvos em: {results_path}")
    
    # Remove checkpoint temporário
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    logger.info("=" * 50)
    logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    logger.info("=" * 50)
    
    return results


def main():
    """Função principal para execução via linha de comando."""
    parser = argparse.ArgumentParser(
        description='Treina modelo LSTM para previsão de preços de ações'
    )
    parser.add_argument(
        '--symbol', 
        type=str, 
        default='PETR4.SA',
        help='Símbolo da ação (default: PETR4.SA)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2018-01-01',
        help='Data inicial para coleta (default: 2018-01-01)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Número máximo de épocas (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamanho do batch (default: 32)'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=60,
        help='Tamanho da sequência LSTM (default: 60)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Diretório para salvar o modelo (default: models)'
    )
    
    args = parser.parse_args()
    
    results = train_model(
        symbol=args.symbol,
        start_date=args.start_date,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_dir=args.model_dir
    )
    
    print("\n" + "=" * 50)
    print("RESUMO DO TREINAMENTO")
    print("=" * 50)
    print(f"Símbolo: {results['symbol']}")
    print(f"Épocas executadas: {results['training_history']['epochs_run']}")
    print(f"MAE: {results['evaluation_metrics']['mae']:.4f}")
    print(f"RMSE: {results['evaluation_metrics']['rmse']:.4f}")
    print(f"MAPE: {results['evaluation_metrics']['mape']:.2f}%")
    

if __name__ == "__main__":
    main()
