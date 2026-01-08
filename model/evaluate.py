"""
Módulo de Avaliação do Modelo
=============================
Implementa métricas de avaliação para o modelo LSTM.

FIAP - Tech Challenge Fase 4

Métricas implementadas:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Classe para avaliação de modelos de previsão de séries temporais.
    
    Implementa as métricas exigidas pelo Tech Challenge:
    - MAE: Erro Absoluto Médio
    - RMSE: Raiz do Erro Quadrático Médio
    - MAPE: Erro Percentual Absoluto Médio
    """
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula o Mean Absolute Error (Erro Absoluto Médio).
        
        MAE = (1/n) * Σ|y_true - y_pred|
        
        Interpretação: Erro médio em unidades da variável alvo (R$).
        Valores menores são melhores.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            
        Returns:
            MAE score
        """
        return float(mean_absolute_error(y_true, y_pred))
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula o Root Mean Square Error (Raiz do Erro Quadrático Médio).
        
        RMSE = √[(1/n) * Σ(y_true - y_pred)²]
        
        Interpretação: Penaliza erros maiores mais severamente que MAE.
        Útil quando erros grandes são particularmente indesejáveis.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            
        Returns:
            RMSE score
        """
        mse = mean_squared_error(y_true, y_pred)
        return float(np.sqrt(mse))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula o Mean Absolute Percentage Error (Erro Percentual Absoluto Médio).
        
        MAPE = (100/n) * Σ|((y_true - y_pred) / y_true)|
        
        Interpretação: Erro percentual médio. Independente de escala.
        - MAPE < 10%: Excelente
        - MAPE 10-20%: Bom
        - MAPE 20-50%: Razoável
        - MAPE > 50%: Impreciso
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            
        Returns:
            MAPE score (em porcentagem)
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Evita divisão por zero
        mask = y_true != 0
        if not mask.any():
            return float('inf')
        
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula o coeficiente de determinação R².
        
        R² = 1 - (SS_res / SS_tot)
        
        Interpretação: Proporção da variância explicada pelo modelo.
        - R² = 1: Modelo perfeito
        - R² = 0: Modelo não melhor que a média
        - R² < 0: Modelo pior que a média
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            
        Returns:
            R² score
        """
        return float(r2_score(y_true, y_pred))
    
    def calculate_all_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcula todas as métricas de avaliação.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            
        Returns:
            Dicionário com todas as métricas
        """
        return {
            "mae": self.mae(y_true, y_pred),
            "rmse": self.rmse(y_true, y_pred),
            "mape": self.mape(y_true, y_pred),
            "r2": self.r2(y_true, y_pred)
        }
    
    def generate_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "LSTM Stock Predictor"
    ) -> str:
        """
        Gera um relatório textual de avaliação.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            model_name: Nome do modelo para o relatório
            
        Returns:
            String formatada com o relatório
        """
        metrics = self.calculate_all_metrics(y_true, y_pred)
        
        # Classificação do MAPE
        mape_val = metrics['mape']
        if mape_val < 10:
            mape_class = "Excelente"
        elif mape_val < 20:
            mape_class = "Bom"
        elif mape_val < 50:
            mape_class = "Razoável"
        else:
            mape_class = "Impreciso"
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              RELATÓRIO DE AVALIAÇÃO DO MODELO                ║
╠══════════════════════════════════════════════════════════════╣
║  Modelo: {model_name:<50} ║
╠══════════════════════════════════════════════════════════════╣
║  MÉTRICAS DE DESEMPENHO                                      ║
╠══════════════════════════════════════════════════════════════╣
║  MAE  (Erro Absoluto Médio):      R$ {metrics['mae']:>10.2f}             ║
║  RMSE (Raiz Erro Quadrático):     R$ {metrics['rmse']:>10.2f}             ║
║  MAPE (Erro Percentual Médio):       {metrics['mape']:>10.2f}%            ║
║  R²   (Coef. Determinação):          {metrics['r2']:>10.4f}              ║
╠══════════════════════════════════════════════════════════════╣
║  INTERPRETAÇÃO                                               ║
╠══════════════════════════════════════════════════════════════╣
║  Classificação MAPE: {mape_class:<40} ║
║                                                              ║
║  • MAE: Em média, as previsões erram R$ {metrics['mae']:.2f}             
║  • RMSE: Erros grandes são penalizados (R$ {metrics['rmse']:.2f})        
║  • R²: O modelo explica {metrics['r2']*100:.1f}% da variância            
╚══════════════════════════════════════════════════════════════╝
"""
        return report
    
    def compare_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_samples: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compara algumas predições com valores reais.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            n_samples: Número de amostras para comparar
            
        Returns:
            Tuple (y_true_sample, y_pred_sample, errors)
        """
        indices = np.linspace(0, len(y_true)-1, n_samples, dtype=int)
        
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        errors = np.abs(y_true_sample - y_pred_sample)
        
        return y_true_sample, y_pred_sample, errors


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    print_report: bool = True
) -> Dict[str, float]:
    """
    Função utilitária para avaliar modelo rapidamente.
    
    Args:
        y_true: Valores reais
        y_pred: Valores preditos
        print_report: Se deve imprimir o relatório
        
    Returns:
        Dicionário com métricas
    """
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_all_metrics(y_true, y_pred)
    
    if print_report:
        report = evaluator.generate_evaluation_report(y_true, y_pred)
        print(report)
    
    return metrics


if __name__ == "__main__":
    # Teste do módulo com dados sintéticos
    np.random.seed(42)
    
    # Simula valores reais e preditos
    y_true = np.array([30.0, 31.5, 32.0, 31.0, 33.0, 34.5, 35.0, 34.0, 36.0, 37.0])
    y_pred = np.array([29.5, 31.8, 32.5, 30.5, 32.5, 34.8, 35.5, 33.5, 35.5, 37.5])
    
    metrics = evaluate_model(y_true, y_pred, print_report=True)
