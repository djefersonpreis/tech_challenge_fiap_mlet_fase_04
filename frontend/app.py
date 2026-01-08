"""
Frontend Streamlit - Stock Price Prediction
============================================
Entry point para aplicação multi-page.

FIAP - Tech Challenge Fase 4
"""

import streamlit as st

# Configuração da página principal
st.set_page_config(
    page_title="Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📈 Sistema de Previsão de Ações")
st.markdown("---")

st.markdown("""
### Bem-vindo ao Sistema de Previsão de Preços de Ações

Este sistema utiliza redes neurais LSTM para prever preços de ações da B3.

#### 📌 Navegação

Use o menu lateral para acessar as funcionalidades:

- **Previsões**: Visualize histórico e previsões de preços
- **Modelos**: Gerencie e treine modelos de previsão

---

### Como funciona?

1. **Coleta de Dados**: Os dados são obtidos em tempo real do Yahoo Finance
2. **Processamento**: Os dados são normalizados e preparados para o modelo
3. **Previsão**: O modelo LSTM processa os dados e gera previsões
4. **Visualização**: Os resultados são exibidos em gráficos interativos

### Símbolos Disponíveis

| Símbolo | Empresa | Setor |
|---------|---------|-------|
| PETR4.SA | Petrobras | Petróleo |
| VALE3.SA | Vale | Mineração |
| ITUB4.SA | Itaú Unibanco | Bancário |
| BBDC4.SA | Bradesco | Bancário |
| ABEV3.SA | Ambev | Bebidas |

> ⚠️ **Aviso**: As previsões são baseadas em modelos de machine learning e 
> **não devem ser usadas como única fonte para decisões de investimento**.

---

*FIAP - Tech Challenge Fase 4 - Machine Learning Engineering*
""")

