"""
Página de Previsões
===================
Visualização de histórico e previsões de preços de ações.

FIAP - Tech Challenge Fase 4
"""

import os
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime

# URL da API
API_URL = os.getenv("API_URL", "http://localhost:8000")


def get_api_health():
    """Verifica o status da API."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_stock_history(symbol: str, days: int = 30):
    """Obtém histórico de preços de uma ação."""
    try:
        response = requests.get(
            f"{API_URL}/predict/stock/{symbol}/history",
            params={"days": days},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": response.json().get("detail", "Erro desconhecido")}
    except Exception as e:
        return {"error": str(e)}


def get_available_models():
    """Obtém lista de modelos disponíveis."""
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Retorna apenas modelos prontos
            return [m["symbol"] for m in data.get("models", []) if m["status"] == "ready"]
        return []
    except Exception:
        return []


def make_prediction(symbol: str, days_ahead: int):
    """Faz uma previsão de preços."""
    try:
        response = requests.post(
            f"{API_URL}/predict/",
            json={"symbol": symbol, "days_ahead": days_ahead},
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def create_price_chart(history_data: list, predictions: list, last_close: float, base_date: str, symbol: str):
    """Cria gráfico interativo com Plotly."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'Preço de {symbol}', 'Volume')
    )
    
    # Dados históricos
    hist_dates = [item["date"] for item in history_data]
    hist_close = [item["close"] for item in history_data]
    hist_volume = [item["volume"] for item in history_data]
    hist_high = [item["high"] for item in history_data]
    hist_low = [item["low"] for item in history_data]
    
    # Candlestick para histórico
    fig.add_trace(
        go.Candlestick(
            x=hist_dates,
            open=[item["open"] for item in history_data],
            high=hist_high,
            low=hist_low,
            close=hist_close,
            name="Histórico",
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Linha de fechamento histórico
    fig.add_trace(
        go.Scatter(
            x=hist_dates,
            y=hist_close,
            mode='lines',
            name='Fechamento',
            line=dict(color='#2196F3', width=2),
            visible='legendonly'
        ),
        row=1, col=1
    )
    
    # Previsões
    if predictions:
        pred_dates = [base_date] + [p["date"] for p in predictions]
        pred_prices = [last_close] + [p["predicted_close"] for p in predictions]
        
        # Linha de previsão
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=pred_prices,
                mode='lines+markers',
                name='Previsão',
                line=dict(color='#FF9800', width=3, dash='dash'),
                marker=dict(size=10, symbol='diamond'),
                hovertemplate='<b>Data:</b> %{x}<br><b>Previsão:</b> R$ %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Área de incerteza (simulada como ±5%)
        upper_bound = [p * 1.05 for p in pred_prices]
        lower_bound = [p * 0.95 for p in pred_prices]
        
        fig.add_trace(
            go.Scatter(
                x=pred_dates + pred_dates[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(255, 152, 0, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Intervalo ±5%',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
    
    # Volume
    colors = ['#26a69a' if hist_close[i] >= (hist_close[i-1] if i > 0 else hist_close[i]) 
              else '#ef5350' for i in range(len(hist_close))]
    
    fig.add_trace(
        go.Bar(
            x=hist_dates,
            y=hist_volume,
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Linha do último preço
    fig.add_hline(
        y=last_close,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Último: R$ {last_close:.2f}",
        annotation_position="right",
        row=1, col=1
    )
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"📈 {symbol} - Histórico e Previsões",
            font=dict(size=20)
        ),
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Preço (R$)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(title_text="Data", row=2, col=1)
    
    return fig


def create_simple_chart(history_data: list, predictions: list, last_close: float, base_date: str, symbol: str):
    """Cria gráfico simplificado de linha."""
    fig = go.Figure()
    
    # Dados históricos
    hist_dates = [item["date"] for item in history_data]
    hist_close = [item["close"] for item in history_data]
    
    # Linha histórica
    fig.add_trace(
        go.Scatter(
            x=hist_dates,
            y=hist_close,
            mode='lines',
            name='Histórico Real',
            line=dict(color='#2196F3', width=2),
            hovertemplate='<b>Data:</b> %{x}<br><b>Preço:</b> R$ %{y:.2f}<extra></extra>'
        )
    )
    
    # Previsões
    if predictions:
        pred_dates = [base_date] + [p["date"] for p in predictions]
        pred_prices = [last_close] + [p["predicted_close"] for p in predictions]
        
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=pred_prices,
                mode='lines+markers',
                name='Previsão',
                line=dict(color='#FF9800', width=3, dash='dash'),
                marker=dict(size=12, symbol='diamond', 
                           line=dict(width=2, color='#FF9800')),
                hovertemplate='<b>Data:</b> %{x}<br><b>Previsão:</b> R$ %{y:.2f}<extra></extra>'
            )
        )
        
        # Área de incerteza
        upper_bound = [p * 1.05 for p in pred_prices]
        lower_bound = [p * 0.95 for p in pred_prices]
        
        fig.add_trace(
            go.Scatter(
                x=pred_dates + pred_dates[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(255, 152, 0, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Intervalo ±5%',
                showlegend=True,
                hoverinfo='skip'
            )
        )
    
    # Ponto de transição
    fig.add_trace(
        go.Scatter(
            x=[base_date],
            y=[last_close],
            mode='markers',
            name='Último Preço Real',
            marker=dict(size=15, color='#4CAF50', symbol='circle',
                       line=dict(width=2, color='white')),
            hovertemplate='<b>Último Preço Real</b><br>Data: %{x}<br>Preço: R$ %{y:.2f}<extra></extra>'
        )
    )
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"📈 {symbol} - Histórico (30 dias) e Previsões",
            font=dict(size=18)
        ),
        xaxis_title="Data",
        yaxis_title="Preço (R$)",
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    return fig


# ============== INTERFACE ==============

st.title("📈 Previsões de Preços")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Busca modelos disponíveis
    available_models = get_available_models()
    
    # Se não conseguir buscar, usa lista padrão
    if not available_models:
        available_models = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA"]
        st.warning("⚠️ Usando lista padrão de símbolos")
    
    # Seleção de símbolo
    symbol = st.selectbox(
        "Símbolo da Ação",
        options=available_models,
        index=0,
        help="Selecione o símbolo da ação para previsão"
    )
    
    # Dias de previsão
    days_ahead = st.slider(
        "Dias de Previsão",
        min_value=1,
        max_value=30,
        value=7,
        help="Número de dias para prever"
    )
    
    # Tipo de gráfico
    chart_type = st.radio(
        "Tipo de Gráfico",
        options=["Simples (Linha)", "Completo (Candlestick)"],
        index=0,
        help="Escolha o tipo de visualização"
    )
    
    st.markdown("---")
    
    # Botão de previsão
    predict_button = st.button("🔮 Fazer Previsão", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # Status da API
    st.header("🔗 Status da API")
    health = get_api_health()
    
    if health.get("status") == "healthy":
        st.success("✅ API Online")
    elif health.get("status") == "degraded":
        st.warning("⚠️ API Degradada")
    else:
        st.error("❌ API Offline")
        if health.get("error"):
            st.caption(f"Erro: {health.get('error')}")

# Área principal
if predict_button:
    # Container de progresso
    progress_container = st.empty()
    
    with progress_container.container():
        col1, col2 = st.columns(2)
        
        with col1:
            with st.spinner("📊 Buscando histórico de 30 dias..."):
                history_result = get_stock_history(symbol, days=30)
        
        with col2:
            with st.spinner("🔮 Gerando previsões..."):
                prediction_result = make_prediction(symbol, days_ahead)
    
    progress_container.empty()
    
    # Verifica erros
    if "error" in history_result:
        st.error(f"❌ Erro ao buscar histórico: {history_result['error']}")
    elif "error" in prediction_result:
        st.error(f"❌ Erro na previsão: {prediction_result.get('error', 'Erro desconhecido')}")
        if "detail" in prediction_result:
            st.caption(prediction_result["detail"])
    else:
        # Sucesso - exibe resultados
        st.subheader(f"📊 Resultado para {symbol}")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        last_close = prediction_result['last_close']
        last_prediction = prediction_result['predictions'][-1]['predicted_close']
        variation = ((last_prediction - last_close) / last_close) * 100
        
        with col1:
            st.metric(
                label="Último Preço Real",
                value=f"R$ {last_close:.2f}"
            )
        
        with col2:
            st.metric(
                label=f"Previsão {days_ahead}º dia",
                value=f"R$ {last_prediction:.2f}",
                delta=f"{variation:+.2f}%"
            )
        
        with col3:
            # Preço mínimo e máximo do histórico
            hist_prices = [item["close"] for item in history_result["data"]]
            st.metric(
                label="Mínima (30d)",
                value=f"R$ {min(hist_prices):.2f}"
            )
        
        with col4:
            st.metric(
                label="Máxima (30d)",
                value=f"R$ {max(hist_prices):.2f}"
            )
        
        st.markdown("---")
        
        # Gráfico
        if chart_type == "Simples (Linha)":
            fig = create_simple_chart(
                history_result["data"],
                prediction_result["predictions"],
                last_close,
                prediction_result["base_date"],
                symbol
            )
        else:
            fig = create_price_chart(
                history_result["data"],
                prediction_result["predictions"],
                last_close,
                prediction_result["base_date"],
                symbol
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabelas lado a lado
        st.markdown("---")
        tab1, tab2 = st.tabs(["📋 Previsões", "📜 Histórico (30 dias)"])
        
        with tab1:
            # Tabela de previsões
            df_pred = pd.DataFrame(prediction_result['predictions'])
            df_pred.columns = ['Data', 'Dia', 'Preço Previsto']
            df_pred['Variação'] = df_pred['Preço Previsto'].apply(
                lambda x: f"{((x - last_close) / last_close) * 100:+.2f}%"
            )
            df_pred['Preço Previsto'] = df_pred['Preço Previsto'].apply(lambda x: f"R$ {x:.2f}")
            
            st.dataframe(df_pred, use_container_width=True, hide_index=True)
        
        with tab2:
            # Tabela de histórico
            df_hist = pd.DataFrame(history_result["data"])
            df_hist = df_hist[['date', 'open', 'high', 'low', 'close', 'volume']]
            df_hist.columns = ['Data', 'Abertura', 'Máxima', 'Mínima', 'Fechamento', 'Volume']
            
            # Formata valores
            for col in ['Abertura', 'Máxima', 'Mínima', 'Fechamento']:
                df_hist[col] = df_hist[col].apply(lambda x: f"R$ {x:.2f}")
            df_hist['Volume'] = df_hist['Volume'].apply(lambda x: f"{x:,.0f}")
            
            # Ordena do mais recente para o mais antigo
            df_hist = df_hist.iloc[::-1]
            
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        
        # Metadados
        st.caption(
            f"Versão do modelo: {prediction_result.get('model_version', 'N/A')} | "
            f"Período histórico: {history_result['start_date']} a {history_result['end_date']} | "
            f"Timestamp: {prediction_result.get('timestamp', 'N/A')}"
        )

else:
    st.info("👈 Configure os parâmetros na barra lateral e clique em **Fazer Previsão**")
    
    # Placeholder com informações
    st.markdown("""
    ### 📊 Visualização de Preços e Previsões
    
    Esta página permite:
    
    1. **Visualizar o histórico** dos últimos 30 dias de negociação
    2. **Gerar previsões** para os próximos dias (1-30)
    3. **Comparar visualmente** dados reais vs previsões
    
    #### Tipos de Gráfico:
    
    - **Simples (Linha)**: Mostra linha de fechamento + previsões
    - **Completo (Candlestick)**: Mostra OHLC + volume + previsões
    
    #### Interpretação:
    
    | Elemento | Descrição |
    |----------|-----------|
    | 🔵 Linha Azul | Preços históricos reais |
    | 🟠 Linha Laranja | Previsões do modelo |
    | 🟠 Círculo Laranja | Ponto de transição (último preço real) |
    | 🟨 Área Sombreada | Intervalo de incerteza (±5%) |
    
    > ⚠️ **Importante**: As previsões são estimativas baseadas em padrões históricos 
    > e não garantem resultados futuros.
    """)

# Footer
st.markdown("---")
st.caption(f"Stock Price Prediction | API: {API_URL}")
