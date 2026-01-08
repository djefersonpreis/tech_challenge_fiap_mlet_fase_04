"""
FastAPI Application - Stock Price Prediction API
=================================================
Ponto de entrada principal da aplicação.

FIAP - Tech Challenge Fase 4
Previsão de preços de ações usando LSTM

Autor: FIAP - Pós-Graduação em Machine Learning Engineering
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.core.model_loader import init_model
from app.api.routes import prediction, health

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduz verbosidade do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação.
    
    - Startup: Carrega o modelo LSTM
    - Shutdown: Cleanup de recursos
    """
    # Startup
    logger.info("=" * 50)
    logger.info("Iniciando Stock Price Prediction API")
    logger.info("=" * 50)
    
    settings = get_settings()
    logger.info(f"Versão: {settings.app_version}")
    logger.info(f"Modelo: {settings.model_path}")
    logger.info(f"Símbolo padrão: {settings.default_symbol}")
    
    # Carrega modelo
    # try:
    #     loader = init_model()
    #     if loader.is_loaded:
    #         logger.info("✓ Modelo carregado com sucesso!")
    #     else:
    #         logger.warning("⚠ Modelo não carregado - API em modo degradado")
    # except Exception as e:
    #     logger.error(f"✗ Erro ao carregar modelo: {e}")
    
    logger.info("=" * 50)
    logger.info("API pronta para receber requisições")
    logger.info("=" * 50)
    
    yield
    
    # Shutdown
    logger.info("Encerrando aplicação...")


def create_app() -> FastAPI:
    """
    Factory function para criar a aplicação FastAPI.
    
    Returns:
        Aplicação FastAPI configurada
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[
            {
                "name": "Predição",
                "description": "Endpoints para previsão de preços de ações"
            },
            {
                "name": "Health & Info",
                "description": "Endpoints de monitoramento e informações"
            }
        ]
    )
    
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Prometheus Instrumentation
    Instrumentator().instrument(app).expose(app)
    
    # Registra rotas
    app.include_router(health.router)
    app.include_router(prediction.router)
    
    return app


# Cria instância da aplicação
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000
    )
