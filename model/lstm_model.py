"""
Arquitetura do Modelo LSTM
==========================
Define a arquitetura da rede neural LSTM para previsão de preços de ações.

FIAP - Tech Challenge Fase 4

Arquitetura:
- 3 camadas LSTM com Dropout para regularização
- Camada Dense intermediária
- Camada de saída com 1 neurônio (preço previsto)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from typing import List, Optional, Tuple
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração para reduzir logs verbosos do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LSTMStockPredictor:
    """
    Modelo LSTM para previsão de preços de ações.
    
    Attributes:
        sequence_length (int): Tamanho da sequência de entrada
        n_features (int): Número de features por timestep
        model (Sequential): Modelo Keras
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 1,
        lstm_units: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2,
        dense_units: int = 25,
        learning_rate: float = 0.001
    ):
        """
        Inicializa o modelo LSTM.
        
        Args:
            sequence_length: Número de timesteps de entrada
            n_features: Número de features (1 para univariado)
            lstm_units: Lista com número de unidades em cada camada LSTM
            dropout_rate: Taxa de dropout para regularização
            dense_units: Unidades na camada Dense intermediária
            learning_rate: Taxa de aprendizado do otimizador Adam
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.model: Optional[Sequential] = None
        self.history = None
        
    def build_model(self) -> Sequential:
        """
        Constrói a arquitetura do modelo LSTM.
        
        Returns:
            Modelo Keras compilado
        """
        logger.info("Construindo modelo LSTM...")
        
        model = Sequential(name="LSTM_Stock_Predictor")
        
        # Camada de entrada
        model.add(Input(shape=(self.sequence_length, self.n_features)))
        
        # Primeira camada LSTM - return_sequences=True para passar sequências
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True,
            name="lstm_1"
        ))
        model.add(Dropout(self.dropout_rate, name="dropout_1"))
        
        # Segunda camada LSTM
        model.add(LSTM(
            units=self.lstm_units[1],
            return_sequences=True,
            name="lstm_2"
        ))
        model.add(Dropout(self.dropout_rate, name="dropout_2"))
        
        # Terceira camada LSTM - return_sequences=False (última LSTM)
        model.add(LSTM(
            units=self.lstm_units[2],
            return_sequences=False,
            name="lstm_3"
        ))
        model.add(Dropout(self.dropout_rate, name="dropout_3"))
        
        # Camada Dense intermediária
        model.add(Dense(
            units=self.dense_units,
            activation='relu',
            name="dense_1"
        ))
        
        # Camada de saída - 1 neurônio para previsão de preço
        model.add(Dense(units=1, name="output"))
        
        # Compilação
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Mean Squared Error
            metrics=['mae']  # Mean Absolute Error
        )
        
        self.model = model
        logger.info("Modelo construído com sucesso!")
        model.summary(print_fn=logger.info)
        
        return model
    
    def get_callbacks(
        self,
        checkpoint_path: Optional[str] = None,
        patience: int = 10
    ) -> List:
        """
        Retorna callbacks para treinamento.
        
        Args:
            checkpoint_path: Caminho para salvar checkpoints
            patience: Paciência para early stopping
            
        Returns:
            Lista de callbacks Keras
        """
        callbacks = []
        
        # Early Stopping - para se não houver melhora
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # Reduz learning rate quando loss estagna
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model Checkpoint - salva melhor modelo
        if checkpoint_path:
            checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        return callbacks
    
    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.1,
        checkpoint_path: Optional[str] = None,
        progress_callback=None
    ):
        """
        Treina o modelo LSTM.
        
        Args:
            X_train: Dados de treino - shape (samples, sequence_length, features)
            y_train: Labels de treino
            X_val: Dados de validação (opcional)
            y_val: Labels de validação (opcional)
            epochs: Número máximo de épocas
            batch_size: Tamanho do batch
            validation_split: Proporção para validação se X_val não fornecido
            checkpoint_path: Caminho para salvar checkpoints
            progress_callback: Objeto com método on_epoch_end(epoch, logs) para reportar progresso
            
        Returns:
            Histórico de treinamento
        """
        if self.model is None:
            self.build_model()
        
        callbacks = self.get_callbacks(
            checkpoint_path=checkpoint_path,
            patience=10
        )
        
        # Adiciona callback de progresso se fornecido
        if progress_callback is not None:
            from tensorflow.keras.callbacks import LambdaCallback
            progress_cb = LambdaCallback(
                on_epoch_end=lambda epoch, logs: progress_callback.on_epoch_end(epoch, logs)
            )
            callbacks.append(progress_cb)
        
        logger.info(f"Iniciando treinamento - Epochs: {epochs}, Batch Size: {batch_size}")
        
        # Configura validação
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = 0.0
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split if validation_data is None else 0.0,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Treinamento concluído!")
        return self.history
    
    def predict(self, X) -> float:
        """
        Realiza predição com o modelo.
        
        Args:
            X: Dados de entrada - shape (samples, sequence_length, features)
            
        Returns:
            Predições do modelo
        """
        if self.model is None:
            raise ValueError("Modelo não treinado. Chame train() ou load_model() primeiro.")
        
        return self.model.predict(X, verbose=0)
    
    def save_model(self, filepath: str) -> None:
        """
        Salva o modelo treinado.
        
        Args:
            filepath: Caminho para salvar (extensão .keras ou .h5)
        """
        if self.model is None:
            raise ValueError("Nenhum modelo para salvar.")
        
        self.model.save(filepath)
        logger.info(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Carrega um modelo previamente salvo.
        
        Args:
            filepath: Caminho do arquivo do modelo
        """
        self.model = load_model(filepath)
        logger.info(f"Modelo carregado de: {filepath}")
    
    def get_model_summary(self) -> dict:
        """
        Retorna informações sobre o modelo.
        
        Returns:
            Dicionário com informações do modelo
        """
        if self.model is None:
            return {"status": "Modelo não construído"}
        
        return {
            "name": self.model.name,
            "total_params": self.model.count_params(),
            "trainable_params": sum([
                tf.keras.backend.count_params(w) 
                for w in self.model.trainable_weights
            ]),
            "layers": len(self.model.layers),
            "input_shape": str(self.model.input_shape),
            "output_shape": str(self.model.output_shape),
            "optimizer": self.model.optimizer.__class__.__name__,
            "loss": self.model.loss
        }


def create_default_model(sequence_length: int = 60) -> LSTMStockPredictor:
    """
    Função factory para criar modelo com configuração padrão.
    
    Args:
        sequence_length: Tamanho da sequência de entrada
        
    Returns:
        Modelo LSTM configurado
    """
    model = LSTMStockPredictor(
        sequence_length=sequence_length,
        n_features=1,
        lstm_units=[128, 64, 32],
        dropout_rate=0.2,
        dense_units=25,
        learning_rate=0.001
    )
    model.build_model()
    return model


if __name__ == "__main__":
    # Teste da arquitetura
    model = create_default_model(sequence_length=60)
    summary = model.get_model_summary()
    
    print("\nInformações do Modelo:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
