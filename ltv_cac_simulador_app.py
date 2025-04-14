import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Carregar dados


uploaded_file = st.file_uploader("Faça upload da base de clientes (.csv)", type="csv")


@st.cache_data
def carregar_dados(file):
    df = pd.read_csv(file)
    df = df.dropna(subset=['LTV', 'lt'])
    df = df[(df["LTV"] > 0) & (df["Deal"] > 500)]
    df = df[df['lt'] > 0]
    
    df['revenues_medio'] = df['LTV'] / df['lt']
    df['custo_operacional'] = df['LTV'] * 0.70
    df['cac'] = 6453
    df['custo_total'] = df['custo_operacional'] + df['cac']
    df['lucro_estimado'] = df['LTV'] - df['custo_total']
    df['viavel'] = df['lucro_estimado'] > 0

    
    X = np.log1p(df[['revenues_medio']])
    kmeans = KMeans(n_clusters=6, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    centroides = kmeans.cluster_centers_.flatten()
    ordem_clusters = centroides.argsort()
    mapa_faixas = {cluster: f'Faixa {i+1}' for i, cluster in enumerate(ordem_clusters)}
    df['faixa_arpu'] = df['cluster'].map(mapa_faixas)

    return df

# 🔁 Verifica se o arquivo foi carregado antes de seguir

# Se não houver arquivo, interrompe
if uploaded_file is not None:
    df = carregar_dados(uploaded_file)
else:
    df = None

# Sidebar - Parâmetros de simulação
st.sidebar.title("Simulador de Viabilidade")
cac_sim = st.sidebar.number_input("CAC (R$) (6.543 nos utlimos meses)", value=6453)
deal_sim = st.sidebar.number_input("Deal médio mensal (R$)", value=5000)
lt_sim = st.sidebar.number_input("Lifetime (meses)", value=7)
margem = st.sidebar.number_input("Margem (%)", value=30, min_value=0, max_value=100) / 100

pagina = st.sidebar.selectbox("Escolha a análise", [
    "Resumo por Faixa de Arpu",
    "Simulador de Viabilidade",
    "Gráfico CAC Payback por Cliente"
])

if pagina == "Resumo por Faixa de Arpu":
    if df is None:
        st.warning("Por favor, envie a base de clientes para visualizar o resumo por faixa.")
        st.stop()
    resumo = df.groupby('faixa_arpu').agg(
        qtd_clientes=('cliente', 'count'),
        receita_total=('LTV', 'sum'),
        lucro_medio=('lucro_estimado', 'mean'),
        lt_medio=('lt', 'mean'),
        deal_medio=('revenues_medio', 'mean'),
        pct_viaveis=('viavel', lambda x: round(x.mean() * 100, 2))
    ).reset_index()

    resumo['deal_min_ideal'] = (df['cac'].mean() / 0.30) / resumo['lt_medio']
    resumo['lt_min_ideal'] = (df['cac'].mean() / 0.30) / resumo['deal_medio']

    st.dataframe(resumo.round(2))

    st.subheader("Distribuição de clientes por faixa")
    hist_data = df['faixa_arpu'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10,6))
    bars = ax.bar(hist_data.index, hist_data.values, color='salmon')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 50, int(yval), ha='center', va='bottom')
    ax.set_title("Clientes por Faixa de arpu")
    ax.set_ylabel("Número de Clientes")
    ax.set_xlabel("Faixa de Arpu")
    st.pyplot(fig)

elif pagina == "Simulador de Viabilidade":
    st.title("Simulador de Viabilidade")

    # LTV com e sem margem
    ltv_bruto = deal_sim * lt_sim
    ltv_sim = ltv_bruto * margem

    # Corrigido: agora o único custo é o CAC
    custo_total_sim = cac_sim
    lucro_sim = ltv_sim - custo_total_sim
    viavel = lucro_sim > 0

    st.metric("LTV Bruto (R$)", round(ltv_bruto, 2))
    st.metric("LTV com Margem (R$)", round(ltv_sim, 2))
    st.metric("Lucro estimado (R$)", round(lucro_sim, 2))
    st.metric("Situação", "Viável" if viavel else "Inviável")

    # Gráfico de payback
    meses = np.arange(max(2, lt_sim + 1))
    lucro_efetivo_mensal = deal_sim * margem
    receita_acumulada = lucro_efetivo_mensal * meses
    lucro_acumulado = receita_acumulada - cac_sim
    payback_index = np.argmax(lucro_acumulado >= 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(meses, lucro_acumulado, label='LTV acumulado', color='green')
    ax.axhline(0, color='black', linestyle='--')
    ax.axvline(payback_index, color='blue', linestyle='--', label='CAC Payback')
    ax.fill_between(meses[:payback_index+1], lucro_acumulado[:payback_index+1], 0, color='red', alpha=0.3, label='CAC')
    ax.fill_between(meses[payback_index:], lucro_acumulado[payback_index:], 0, color='green', alpha=0.3, label='Retorno')
    ax.set_title("CAC Payback Simulado")
    ax.set_xlabel("Meses")
    ax.set_ylabel("Lucro Acumulado (R$)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

elif pagina == "Gráfico CAC Payback por Cliente":
    if df is None:
        st.warning("Por favor, envie a base de clientes para visualizar o gráfico por cliente.")
        st.stop()
    cliente_nome = st.selectbox("Selecione o cliente:", df['cliente'].unique())
    cliente = df[df["cliente"] == cliente_nome].iloc[0]

    lt = int(cliente["lt"])
    cac = cliente["cac"]
    revenues_mensais = cliente["revenues_medio"]

    meses = np.arange(lt + 1)
    lucro_efetivo_mensal = revenues_mensais * margem
    receita_acumulada = lucro_efetivo_mensal * meses
    lucro_acumulado = receita_acumulada - cac
    payback_index = np.argmax(lucro_acumulado >= 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(meses, lucro_acumulado, label='LTV acumulado', color='green')
    ax.axhline(0, color='black', linestyle='--')
    ax.axvline(payback_index, color='blue', linestyle='--', label='CAC Payback')
    ax.fill_between(meses[:payback_index+1], lucro_acumulado[:payback_index+1], 0, color='red', alpha=0.3, label='CAC')
    ax.fill_between(meses[payback_index:], lucro_acumulado[payback_index:], 0, color='green', alpha=0.3, label='Retorno')
    ax.set_title(f"LTV vs CAC Payback - {cliente['cliente'][:50]}...")
    ax.set_xlabel("Meses")
    ax.set_ylabel("Lucro Acumulado (R$)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
