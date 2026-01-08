# Stock Price Prediction API - LSTM

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docs.docker.com/compose/)

> **FIAP - Tech Challenge Fase 4**  
> Pós-Graduação em Machine Learning Engineering

API RESTful para previsão de preços de fechamento de ações utilizando redes neurais **Long Short Term Memory (LSTM)**.

---

## Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Arquitetura](#arquitetura)
- [Tecnologias](#tecnologias)
- [Instalação](#instalação)
- [Uso](#uso)
- [Endpoints da API](#endpoints-da-api)
- [Modelo LSTM](#modelo-lstm)
- [Monitoramento](#monitoramento)
- [Testes](#testes)
- [Estrutura do Projeto](#estrutura-do-projeto)

---

## Sobre o Projeto

Este projeto implementa uma pipeline completa de **Machine Learning** para previsão de preços de ações, desde a coleta de dados até o deploy em produção com monitoramento.

### Funcionalidades

- **Coleta automática de dados** via Yahoo Finance (yfinance)
- **Modelo LSTM** com arquitetura Bidirectional + BatchNorm + L2 Regularization
- **API RESTful** com FastAPI para previsões em tempo real
- **Containerização** com Docker e Docker Compose
- **Monitoramento** com Prometheus e Grafana
- **12 features técnicas** (RSI, MACD, Bollinger Bands, ATR, etc.)

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                         Cliente                                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI (Port 8000)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   /predict  │  │   /health   │  │      /metrics           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Prediction Service                            │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │  Data Collector │  │   Preprocessor   │  │  LSTM Model    │  │
│  │   (yfinance)    │  │   (Scaler)       │  │  (TensorFlow)  │  │
│  └─────────────────┘  └──────────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Monitoring Stack                            │
│  ┌─────────────────────┐        ┌────────────────────────────┐  │
│  │    Prometheus       │◄──────►│        Grafana             │  │
│  │    (Port 9090)      │        │      (Port 3000)           │  │
│  └─────────────────────┘        └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tecnologias

### Backend & ML
| Tecnologia | Versão | Descrição |
|------------|--------|-----------|
| Python | 3.11 | Linguagem principal |
| FastAPI | 0.109.0 | Framework web assíncrono |
| TensorFlow | 2.20.0 | Deep Learning framework |
| Pandas | 2.1.4 | Manipulação de dados |
| Scikit-learn | 1.4.0 | Pré-processamento |
| yfinance | 0.2.59 | Coleta de dados financeiros |

### Infraestrutura
| Tecnologia | Descrição |
|------------|-----------|
| Docker | Containerização |
| Docker Compose | Orquestração de containers |
| Prometheus | Coleta de métricas |
| Grafana | Visualização de métricas |

---

## Instalação

### Pré-requisitos

- Python 3.11+
- Docker e Docker Compose (para deploy containerizado)
- pip ou conda

### Instalação Local

1. **Clone o repositório**
```bash
git clone https://github.com/seu-usuario/fiap-mlet-fase4.git
cd fiap-mlet-fase4
```

2. **Crie um ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Configure as variáveis de ambiente**
```bash
cp .env.example .env
# Edite o arquivo .env conforme necessário
```

5. **Execute a API**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Deploy com Docker

```bash
# Build e execução com Docker Compose
docker-compose up -d --build

# Verificar status dos containers
docker-compose ps

# Ver logs
docker-compose logs -f api
```

Serviços disponíveis:
- **API**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

---

## Uso

### Exemplo de Previsão via cURL

```bash
# Previsão para PETR4.SA (Petrobras) - 5 dias
curl -X POST "http://localhost:8000/predict/" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "PETR4.SA", "days_ahead": 5}'
```

### Exemplo de Resposta

```json
{
  "symbol": "PETR4.SA",
  "predictions": [
    {"date": "2026-01-09", "predicted_price": 35.42},
    {"date": "2026-01-10", "predicted_price": 35.68},
    {"date": "2026-01-13", "predicted_price": 35.91},
    {"date": "2026-01-14", "predicted_price": 36.15},
    {"date": "2026-01-15", "predicted_price": 36.38}
  ],
  "last_known_price": 34.58,
  "model_version": "2.0_improved",
  "generated_at": "2026-01-08T12:00:00"
}
```

### Exemplo com Python

```python
import requests

url = "http://localhost:8000/predict/"
payload = {
    "symbol": "VALE3.SA",
    "days_ahead": 7
}

response = requests.post(url, json=payload)
predictions = response.json()

for pred in predictions["predictions"]:
    print(f"{pred['date']}: R$ {pred['predicted_price']:.2f}")
```

---

## Endpoints da API

### Predição

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `POST` | `/predict/` | Previsão de preços para um símbolo |
| `POST` | `/predict/custom` | Previsão com dados históricos customizados |

### Health & Info

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/` | Informações básicas da API |
| `GET` | `/health` | Status de saúde da API |
| `GET` | `/model/info` | Informações do modelo LSTM |
| `GET` | `/metrics` | Métricas Prometheus |

### Documentação Interativa

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Modelo LSTM

### Arquitetura

```
Input (60 timesteps × 12 features)
         │
         ▼
┌─────────────────────────┐
│  Bidirectional LSTM     │  128 units
│  + BatchNormalization   │
│  + Dropout (0.3)        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Bidirectional LSTM     │  64 units
│  + BatchNormalization   │
│  + Dropout (0.3)        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  LSTM                   │  32 units
│  + Dropout (0.3)        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Dense (25) + ReLU      │
│  Dense (1) - Output     │
└─────────────────────────┘
```

### Features Utilizadas

| Feature | Descrição |
|---------|-----------|
| Close | Preço de fechamento |
| RSI | Relative Strength Index |
| MACD | Moving Average Convergence Divergence |
| MACD_Hist | Histograma MACD |
| BB_Position | Posição nas Bandas de Bollinger |
| BB_Width | Largura das Bandas de Bollinger |
| ATR | Average True Range |
| Volatility | Volatilidade |
| Momentum_10 | Momentum de 10 períodos |
| Volume_Norm | Volume normalizado |
| ROC | Rate of Change |
| Log_Return | Retorno logarítmico |

### Métricas do Modelo

| Métrica | Valor |
|---------|-------|
| MAE | 0.828 |
| RMSE | 1.052 |
| MAPE | 2.65% |
| R² | 0.548 |

### Treinamento

O modelo foi treinado com dados históricos de **PETR4.SA** (Petrobras):
- **Período**: 2018-01-01 a 2026-01-05
- **Total de registros**: 1.960
- **Split**: 80% treino / 20% teste
- **Epochs**: 89 (early stopping)
- **Loss function**: Huber Loss

---

## Monitoramento

### Prometheus

Métricas coletadas:
- Requisições por segundo
- Latência de resposta
- Erros HTTP
- Uso de CPU/Memória

Acesse: http://localhost:9090

### Grafana

Dashboard pré-configurado com:
- Taxa de requisições
- Tempo de resposta (p50, p95, p99)
- Status de saúde da API
- Erros por endpoint

Acesse: http://localhost:3000
- **Usuário**: admin
- **Senha**: admin

---

## Testes

### Executar testes

```bash
# Instalar dependências de desenvolvimento
pip install -r requirements-dev.txt

# Executar todos os testes
pytest

# Com cobertura
pytest --cov=app --cov-report=html

# Testes específicos
pytest tests/test_api.py -v
pytest tests/test_model.py -v
```

---

## Estrutura do Projeto

```
fiap-mlet-fase4/
├── app/                          # Código da API
│   ├── api/
│   │   ├── routes/               # Endpoints
│   │   │   ├── health.py         # Health checks
│   │   │   └── prediction.py     # Rotas de predição
│   │   └── schemas/              # Schemas Pydantic
│   │       └── prediction.py
│   ├── core/
│   │   └── model_loader.py       # Carregador do modelo
│   ├── services/
│   │   └── prediction_service.py # Lógica de predição
│   ├── config.py                 # Configurações
│   └── main.py                   # Entrypoint FastAPI
├── model/                        # Código de ML
│   ├── data_collector.py         # Coleta de dados
│   ├── preprocessor.py           # Pré-processamento
│   ├── lstm_model.py             # Arquitetura LSTM
│   ├── train.py                  # Script de treinamento
│   └── evaluate.py               # Avaliação do modelo
├── models/                       # Artefatos treinados
│   ├── lstm_model.keras          # Modelo salvo
│   ├── feature_columns.json      # Colunas utilizadas
│   └── training_results.json     # Métricas de treino
├── monitoring/                   # Stack de monitoramento
│   ├── grafana/
│   │   └── provisioning/
│   └── prometheus/
│       └── prometheus.yml
├── notebooks/                    # Jupyter notebooks
│   └── 01_exploracao_e_treinamento.ipynb
├── tests/                        # Testes automatizados
│   ├── test_api.py
│   └── test_model.py
├── docker-compose.yml            # Orquestração Docker
├── Dockerfile                    # Imagem da API
├── Dockerfile.train              # Imagem de treino
├── requirements.txt              # Dependências prod
├── requirements-dev.txt          # Dependências dev
└── README.md                     # Este arquivo
```

---

## Retreinamento do Modelo

Para retreinar o modelo com novos dados:

```bash
# Via Docker
docker build -f Dockerfile.train -t stock-lstm-train .
docker run -v $(pwd)/models:/app/models stock-lstm-train

# Localmente
python -m model.train
```

---

## Variáveis de Ambiente

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `MODEL_PATH` | Caminho do modelo .keras | `/app/models/lstm_model.keras` |
| `SCALER_PATH` | Caminho do scaler | `/app/models/scaler.pkl` |
| `DEFAULT_SYMBOL` | Símbolo padrão | `PETR4.SA` |
| `MAX_PREDICTION_DAYS` | Máximo de dias para previsão | `30` |