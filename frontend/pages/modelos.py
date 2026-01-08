"""
Página de Gerenciamento de Modelos
==================================
Controle de modelos existentes e treinamento de novos modelos.

FIAP - Tech Challenge Fase 4
"""

import os
import time
import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# URL da API
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Constantes
POLLING_INTERVAL = 5  # segundos


def get_models_list():
    """Obtém lista de modelos."""
    try:
        response = requests.get(f"{API_URL}/models", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"error": response.json().get("detail", "Erro desconhecido")}
    except Exception as e:
        return {"error": str(e)}


def get_model_details(symbol: str):
    """Obtém detalhes de um modelo específico."""
    try:
        response = requests.get(f"{API_URL}/models/{symbol}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"error": response.json().get("detail", "Erro desconhecido")}
    except Exception as e:
        return {"error": str(e)}


def get_training_status(symbol: str):
    """Obtém status de treinamento de um modelo."""
    try:
        response = requests.get(f"{API_URL}/models/{symbol}/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def get_queue_status():
    """Obtém status da fila de treinamento."""
    try:
        response = requests.get(f"{API_URL}/models/queue/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"error": response.json().get("detail", "Erro desconhecido")}
    except Exception as e:
        return {"error": str(e)}


def start_training(symbol: str, epochs: int, start_date: str, batch_size: int):
    """Inicia treinamento de um modelo."""
    try:
        response = requests.post(
            f"{API_URL}/models/{symbol}/train",
            json={
                "epochs": epochs,
                "start_date": start_date,
                "batch_size": batch_size
            },
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def reload_model(symbol: str):
    """Recarrega um modelo na memória."""
    try:
        response = requests.post(f"{API_URL}/models/{symbol}/reload", timeout=30)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def format_metrics(metrics: dict) -> str:
    """Formata métricas para exibição."""
    if not metrics:
        return "N/A"
    
    parts = []
    if metrics.get("mae"):
        parts.append(f"MAE: {metrics['mae']:.4f}")
    if metrics.get("rmse"):
        parts.append(f"RMSE: {metrics['rmse']:.4f}")
    if metrics.get("mape"):
        parts.append(f"MAPE: {metrics['mape']:.2f}%")
    if metrics.get("r2"):
        parts.append(f"R²: {metrics['r2']:.4f}")
    
    return " | ".join(parts) if parts else "N/A"


def render_model_card(model: dict):
    """Renderiza card de um modelo."""
    symbol = model["symbol"]
    status = model["status"]
    
    # Cor do status
    status_colors = {
        "ready": "🟢",
        "training": "🟡",
        "queued": "🟠",
        "failed": "🔴"
    }
    status_icon = status_colors.get(status, "⚪")
    
    with st.container():
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col1:
            st.markdown(f"### {status_icon} {symbol}")
            st.caption(f"Status: **{status.upper()}**")
        
        with col2:
            if model.get("metrics"):
                st.markdown("**Métricas:**")
                metrics = model["metrics"]
                
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("MAE", f"{metrics.get('mae', 0):.4f}" if metrics.get('mae') else "N/A")
                with metric_cols[1]:
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}" if metrics.get('rmse') else "N/A")
                with metric_cols[2]:
                    st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%" if metrics.get('mape') else "N/A")
                with metric_cols[3]:
                    st.metric("R²", f"{metrics.get('r2', 0):.4f}" if metrics.get('r2') else "N/A")
            else:
                st.caption("Métricas não disponíveis")
        
        with col3:
            if model.get("trained_at"):
                trained_date = model["trained_at"][:10] if model["trained_at"] else "N/A"
                st.caption(f"Treinado em: {trained_date}")
            
            if model.get("is_loaded_in_memory"):
                st.success("✅ Em memória")
            else:
                if status == "ready":
                    if st.button("🔄 Carregar", key=f"reload_{symbol}"):
                        with st.spinner("Carregando..."):
                            result = reload_model(symbol)
                        if "error" not in result:
                            st.success("Modelo carregado!")
                            st.rerun()
                        else:
                            st.error(f"Erro: {result['error']}")
        
        st.markdown("---")


def render_training_progress(job: dict):
    """Renderiza barra de progresso do treinamento."""
    symbol = job["symbol"]
    progress = job.get("progress", 0)
    epochs_current = job.get("epochs_current", 0)
    epochs_total = job.get("epochs_total", 0)
    status = job["status"]
    
    st.markdown(f"### 🔄 Treinando: {symbol}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.progress(progress / 100, text=f"Epoch {epochs_current}/{epochs_total} ({progress}%)")
    
    with col2:
        st.caption(f"Status: {status}")
    
    if job.get("started_at"):
        started = job["started_at"][:19].replace("T", " ")
        st.caption(f"Iniciado em: {started}")
    
    return status in ["training", "queued"]


# ============== INTERFACE ==============

st.title("🤖 Gerenciamento de Modelos")
st.markdown("---")

# Tabs principais
tab1, tab2, tab3 = st.tabs(["📋 Modelos Disponíveis", "🚀 Treinar Novo Modelo", "📊 Fila de Treinamento"])

# ==========================================
# TAB 1: Lista de Modelos
# ==========================================
with tab1:
    st.subheader("Modelos Disponíveis")
    
    # Botão de atualizar
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Atualizar", key="refresh_models"):
            st.rerun()
    
    # Busca modelos
    models_data = get_models_list()
    
    if "error" in models_data:
        st.error(f"❌ Erro ao carregar modelos: {models_data['error']}")
    else:
        models = models_data.get("models", [])
        
        if not models:
            st.info("Nenhum modelo disponível. Use a aba 'Treinar Novo Modelo' para criar um.")
        else:
            # Resumo
            st.markdown(f"""
            **Total:** {models_data.get('total', 0)} modelos | 
            **Prontos:** {models_data.get('ready_count', 0)} | 
            **Em treinamento:** {models_data.get('training_count', 0)}
            """)
            
            st.markdown("---")
            
            # Lista de modelos
            for model in models:
                render_model_card(model)

# ==========================================
# TAB 2: Treinar Novo Modelo
# ==========================================
with tab2:
    st.subheader("Treinar Novo Modelo")
    
    st.markdown("""
    Configure os parâmetros e inicie o treinamento de um novo modelo.
    O treinamento será adicionado à fila e executado em background.
    """)
    
    with st.form("training_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Símbolo
            symbol_input = st.text_input(
                "Símbolo da Ação",
                value="",
                placeholder="Ex: VALE3.SA, ITUB4.SA",
                help="Digite o símbolo da ação no formato Yahoo Finance (ex: PETR4.SA)"
            ).upper()
            
            # Data inicial
            start_date = st.date_input(
                "Data Inicial dos Dados",
                value=datetime(2018, 1, 1),
                min_value=datetime(2010, 1, 1),
                max_value=datetime.now(),
                help="Data inicial para coleta de dados históricos"
            )
        
        with col2:
            # Epochs
            epochs = st.slider(
                "Número de Epochs",
                min_value=10,
                max_value=500,
                value=150,
                step=10,
                help="Número de épocas de treinamento (mais epochs = mais tempo, possível melhor resultado)"
            )
            
            # Batch size
            batch_size = st.select_slider(
                "Batch Size",
                options=[8, 16, 32, 64, 128],
                value=32,
                help="Tamanho do batch para treinamento"
            )
        
        st.markdown("---")
        
        # Estimativa de tempo
        estimated_time = (epochs / 50) * 30  # ~30 segundos por 50 epochs (aproximação)
        st.info(f"⏱️ Tempo estimado: ~{int(estimated_time)} segundos (pode variar)")
        
        # Botão de submit
        submitted = st.form_submit_button("🚀 Iniciar Treinamento", type="primary", use_container_width=True)
        
        if submitted:
            if not symbol_input:
                st.error("❌ Por favor, informe o símbolo da ação")
            elif not symbol_input.endswith(".SA"):
                st.warning("⚠️ Símbolos da B3 geralmente terminam com '.SA' (ex: PETR4.SA)")
                # Permite continuar mesmo assim
                with st.spinner(f"Iniciando treinamento para {symbol_input}..."):
                    result = start_training(
                        symbol_input,
                        epochs,
                        start_date.strftime("%Y-%m-%d"),
                        batch_size
                    )
                
                if "error" in result:
                    st.error(f"❌ Erro: {result['error']}")
                elif "detail" in result:
                    st.error(f"❌ Erro: {result['detail']}")
                else:
                    st.success(f"✅ {result.get('message', 'Treinamento iniciado!')}")
                    st.info("Acompanhe o progresso na aba 'Fila de Treinamento'")
            else:
                with st.spinner(f"Iniciando treinamento para {symbol_input}..."):
                    result = start_training(
                        symbol_input,
                        epochs,
                        start_date.strftime("%Y-%m-%d"),
                        batch_size
                    )
                
                if "error" in result:
                    st.error(f"❌ Erro: {result['error']}")
                elif "detail" in result:
                    st.error(f"❌ Erro: {result['detail']}")
                else:
                    st.success(f"✅ {result.get('message', 'Treinamento iniciado!')}")
                    st.info("Acompanhe o progresso na aba 'Fila de Treinamento'")

# ==========================================
# TAB 3: Fila de Treinamento
# ==========================================
with tab3:
    st.subheader("Fila de Treinamento")
    
    # Placeholder para auto-refresh
    queue_placeholder = st.empty()
    
    # Busca status da fila
    queue_data = get_queue_status()
    
    if "error" in queue_data:
        st.error(f"❌ Erro ao carregar fila: {queue_data['error']}")
    else:
        with queue_placeholder.container():
            # Status geral
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                worker_status = "🟢 Ativo" if queue_data.get("worker_running") else "🔴 Inativo"
                st.metric("Worker", worker_status)
            
            with col2:
                st.metric("Na Fila", queue_data.get("queued_count", 0))
            
            with col3:
                st.metric("Concluídos", queue_data.get("completed_count", 0))
            
            with col4:
                st.metric("Falhas", queue_data.get("failed_count", 0))
            
            st.markdown("---")
            
            # Jobs
            jobs = queue_data.get("jobs", [])
            
            if not jobs:
                st.info("Nenhum job na fila. Inicie um treinamento na aba anterior.")
            else:
                # Job atual (em treinamento)
                current_job = queue_data.get("current_job")
                training_in_progress = False
                
                for job in jobs:
                    status = job["status"]
                    symbol = job["symbol"]
                    
                    if status == "training":
                        training_in_progress = render_training_progress(job)
                    
                    elif status == "queued":
                        st.markdown(f"### 🟠 Na fila: {symbol}")
                        st.caption(f"Aguardando processamento...")
                        if job.get("parameters"):
                            params = job["parameters"]
                            st.caption(f"Epochs: {params.get('epochs', 'N/A')} | Start: {params.get('start_date', 'N/A')}")
                    
                    elif status == "completed":
                        st.markdown(f"### ✅ Concluído: {symbol}")
                        if job.get("result_metrics"):
                            st.caption(format_metrics(job["result_metrics"]))
                        if job.get("completed_at"):
                            completed = job["completed_at"][:19].replace("T", " ")
                            st.caption(f"Finalizado em: {completed}")
                    
                    elif status == "failed":
                        st.markdown(f"### ❌ Falhou: {symbol}")
                        if job.get("error_message"):
                            st.error(f"Erro: {job['error_message']}")
                    
                    st.markdown("---")
                
                # Auto-refresh se houver treinamento em andamento
                if training_in_progress:
                    st.info(f"🔄 Atualizando automaticamente a cada {POLLING_INTERVAL} segundos...")
                    time.sleep(POLLING_INTERVAL)
                    st.rerun()

# Footer
st.markdown("---")
st.caption(f"Gerenciamento de Modelos | API: {API_URL}")
