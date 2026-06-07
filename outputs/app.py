import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# Page configuration
st.set_page_config(
    page_title="Sistema Analítico Ames Housing",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling and styling consistent with advanced computer science projects
st.markdown("""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling - Keeping it intentionally dark for a modern look */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        color: #f8fafc;
    }
    [data-testid="stSidebar"] *, [data-testid="stSidebar"] span {
        color: #f8fafc !important;
    }
    
    /* Main container and headers */
    .stApp {
        background-color: var(--background-color);
    }
    
    .header-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border-left: 8px solid #3b82f6;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    .header-container h1 {
        color: white !important;
        font-weight: 800;
        font-size: 2.25rem;
        margin: 0;
        letter-spacing: -0.025em;
    }
    
    .header-container p {
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-size: 1.1rem;
        color: #94a3b8;
    }
    
    /* KPI Metrics Cards */
    .kpi-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .kpi-card {
        flex: 1;
        background-color: var(--secondary-background-color);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    
    .kpi-title {
        font-size: 0.8rem;
        color: var(--text-color);
        opacity: 0.7;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--text-color);
        margin-top: 0.5rem;
    }
    
    /* Section Cards */
    .section-card {
        background-color: var(--secondary-background-color);
        border-radius: 12px;
        padding: 1.75rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        color: var(--text-color);
    }
    
    .section-card p,
    .section-card ul,
    .section-card li,
    .section-card span,
    .section-card b,
    .section-card i {
        color: var(--text-color) !important;
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 1rem;
        border-bottom: 2px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 0.5rem;
    }
    
    /* Tabs styling for better visibility across all browsers */
    button[data-baseweb="tab"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-bottom: none !important;
        border-radius: 8px 8px 0 0 !important;
        margin-right: 4px !important;
        padding: 0.5rem 1rem !important;
        color: var(--text-color) !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        opacity: 0.7;
    }
    
    button[data-baseweb="tab"]:hover {
        opacity: 1;
        background-color: var(--background-color) !important;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
        border-color: #3b82f6 !important;
        opacity: 1;
    }
    
    div[data-baseweb="tab-list"] {
        gap: 0 !important;
        border-bottom: 2px solid rgba(128, 128, 128, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

import os

# Data loading
@st.cache_data
def load_raw_data():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, '..', 'data')
        
        df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        return df_train, df_test
    except Exception as e:
        st.error(f"Erro ao ler os arquivos de dados: {e}")
        return None, None

df_train, df_test = load_raw_data()

if df_train is None:
    st.stop()

# --- MODEL TRAINING AND PROCESSING ON WHOLE DATASET (CACHED) ---
@st.cache_resource
def process_and_model_whole_dataset(_df):
    df = _df.copy()
    
    # 1. Tratar valores nulos para numéricos
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Id' in num_cols:
        num_cols.remove('Id')
    if 'SalePrice' in num_cols:
        num_cols.remove('SalePrice')
        
    imputer_num = SimpleImputer(strategy='median')
    df[num_cols] = imputer_num.fit_transform(df[num_cols])
    
    # Tratar valores nulos para categóricos
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    
    # 2. Codificação categórica (Label Encoding)
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        
    # Separação
    X = df.drop(['Id', 'SalePrice'], axis=1)
    y = df['SalePrice']
    
    # Normalização
    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=X.columns)
    
    # SPLIT e treino: Regressão
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg_model.fit(X_train, y_train)
    y_pred_val = reg_model.predict(X_val)
    
    # SPLIT e treino: Classificação (acima/abaixo da mediana de SalePrice)
    mediana_preco = y.median()
    y_class = (y > mediana_preco).astype(int)
    Xc_train, Xc_val, yc_train, yc_val = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)
    
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_model.fit(Xc_train, yc_train)
    yc_pred_val = clf_model.predict(Xc_val)
    
    # KMeans Clustering (n=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters
    
    # Map cluster numbers to meaningful labels (Populares, Médios, Luxo) based on average SalePrice
    cluster_means = df.groupby('Cluster')['SalePrice'].mean().sort_values()
    cluster_mapping = {
        cluster_means.index[0]: "Populares",
        cluster_means.index[1]: "Médios",
        cluster_means.index[2]: "Luxo"
    }
    df['Perfil'] = df['Cluster'].map(cluster_mapping)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    
    # Isolation Forest (Outlier detection)
    # Target exactly 73 outliers from the dataset (which corresponds to contamination ~ 0.05)
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outliers = iso_forest.fit_predict(X_scaled)
    df['Outlier'] = outliers # 1 for normal, -1 for outlier
    
    return {
        'df_processed': df,
        'X_scaled': X_scaled,
        'y': y,
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'y_pred_val': y_pred_val,
        'reg_model': reg_model,
        'mediana_preco': mediana_preco,
        'y_class': y_class,
        'Xc_train': Xc_train,
        'Xc_val': Xc_val,
        'yc_train': yc_train,
        'yc_val': yc_val,
        'yc_pred_val': yc_pred_val,
        'clf_model': clf_model,
        'kmeans': kmeans,
        'pca': pca,
        'iso_forest': iso_forest,
        'scaler': scaler
    }

results = process_and_model_whole_dataset(df_train)
df_processed = results['df_processed']

# --- SIDEBAR: DYNAMIC FILTERS ---
st.sidebar.title("🛠️ Painel de Filtros")
st.sidebar.markdown("Filtre as propriedades exibidas nos gráficos e métricas em tempo real:")

# Neighborhood filter
all_neighborhoods = sorted(df_train['Neighborhood'].unique().tolist())
select_all_neigh = st.sidebar.checkbox("Selecionar Todos os Bairros", value=True)

if select_all_neigh:
    selected_neighborhoods = all_neighborhoods
else:
    selected_neighborhoods = st.sidebar.multiselect(
        "Selecione as Vizinhanças (Neighborhood):",
        options=all_neighborhoods,
        default=all_neighborhoods[:3]
    )

# OverallQual filter (Slider)
min_qual = int(df_train['OverallQual'].min())
max_qual = int(df_train['OverallQual'].max())
selected_qual = st.sidebar.slider(
    "Qualidade Geral (OverallQual):",
    min_value=min_qual,
    max_value=max_qual,
    value=(min_qual, max_qual)
)

# Apply dynamic filters to df_train first (to match string-based categories properly)
df_filtered_raw = df_train[
    (df_train['Neighborhood'].isin(selected_neighborhoods)) &
    (df_train['OverallQual'] >= selected_qual[0]) &
    (df_train['OverallQual'] <= selected_qual[1])
]

# Ensure dashboard doesn't crash on empty filtered dataframe
if df_filtered_raw.empty:
    st.sidebar.warning("Nenhum imóvel corresponde aos filtros!")
    st.warning("⚠️ Nenhum imóvel corresponde aos filtros selecionados. Por favor, ajuste as opções na barra lateral.")
    st.stop()

# Then, df_filtered is df_processed indexed by df_filtered_raw.index (maintaining PCA/Cluster data)
df_filtered = df_processed.loc[df_filtered_raw.index]

# --- TOP HEADER ---
st.markdown("""
<div class="header-container">
    <h1>Sistema Analítico Ames Housing</h1>
    <p>Análise de Dados Aplicada à Computação — Projeto Acadêmico Avançado</p>
</div>
""", unsafe_allow_html=True)

# --- KPI METRICS CARDS (Top of the Page) ---
kpi_cols = st.columns(3)
with kpi_cols[0]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Imóveis Filtrados / Total</div>
        <div class="kpi-value">{df_filtered.shape[0]} <span style="font-size:1.1rem; font-weight:400; color:#64748b;">/ {df_train.shape[0]}</span></div>
    </div>
    """, unsafe_allow_html=True)
with kpi_cols[1]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Preço Médio (Filtrado)</div>
        <div class="kpi-value">${df_filtered['SalePrice'].mean():,.2f}</div>
    </div>
    """, unsafe_allow_html=True)
with kpi_cols[2]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Preço Máximo (Filtrado)</div>
        <div class="kpi-value">${df_filtered['SalePrice'].max():,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

# --- INTERACTIVE TABS ---
tab1, tab2, tab3 = st.tabs([
    "📊 Análise Exploratória & Engenharia de Dados",
    "📈 Aprendizado Supervisionado (Modelos Clássicos)",
    "🧩 Aprendizado Não Supervisionado & Visão Espacial"
])

# ----------------- ABA 1: EDA & ENGENHARIA DE DADOS -----------------
with tab1:
    st.write("### Análise Exploratória & Engenharia de Dados")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.write("<div class='section-title'>Distribuição do Preço de Venda (SalePrice)</div>", unsafe_allow_html=True)
        fig_hist = px.histogram(
            df_filtered,
            x='SalePrice',
            nbins=40,
            marginal='box',
            color_discrete_sequence=['#3b82f6'],
            labels={'SalePrice': 'Preço de Venda ($)'}
        )
        fig_hist.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.write("<div class='section-title'>Matriz de Correlação Interativa</div>", unsafe_allow_html=True)
        
        # Select numeric variables that exist in the original training data (excluding Id)
        orig_num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        if 'Id' in orig_num_cols:
            orig_num_cols.remove('Id')
            
        # Compute correlation on the filtered subset using original numeric features
        corr_mat = df_filtered[orig_num_cols].corr()
        # Find top 10 variables correlated with SalePrice
        top_features = corr_mat['SalePrice'].abs().sort_values(ascending=False).index[:10]
        corr_mat_top = corr_mat.loc[top_features, top_features]
        
        fig_heat = px.imshow(
            corr_mat_top,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig_heat.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Summary of Null Value Imputations
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Tratamento de Dados & Engenharia de Atributos (Pipeline)</div>
        <p>No desenvolvimento do pipeline computacional para modelagem do dataset, as seguintes transformações foram realizadas:</p>
        <ul>
            <li><b>Imputação de Valores Nulos (Numéricos):</b> Preenchidos automaticamente utilizando a <b>mediana</b> de cada coluna (para evitar influência de valores anômalos/outliers).</li>
            <li><b>Imputação de Valores Nulos (Categóricos):</b> Preenchidos com a <b>moda</b> (valor mais frequente) de cada categoria.</li>
            <li><b>Codificação Qualitativa:</b> As variáveis categóricas textuais foram convertidas em valores numéricos inteiros ordinais utilizando <i>Label Encoder</i>.</li>
            <li><b>Padronização de Escala (Normalização):</b> Aplicação de <i>StandardScaler</i> em todas as variáveis explicativas, centralizando-as na média zero e variância unitária (essencial para PCA, K-Means e modelos de regressão/classificação).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ----------------- ABA 2: APRENDIZADO SUPERVISIONADO -----------------
with tab2:
    st.write("### Modelos Supervisionados (Random Forest)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-card" style="height:100%;">', unsafe_allow_html=True)
        st.write("<div class='section-title'>Modelagem de Regressão (Previsão de Preço)</div>", unsafe_allow_html=True)
        
        # Calculate regression metrics specifically on the filtered subset of the validation set
        X_val = results['X_val']
        y_val = results['y_val']
        y_pred_val = results['y_pred_val']
        reg_model = results['reg_model']
        
        # Identify validation set rows that fall into our current filter selection
        filtered_val_idx = df_filtered.index.intersection(X_val.index)
        
        if len(filtered_val_idx) > 2:
            y_val_filt = y_val.loc[filtered_val_idx]
            X_val_filt = X_val.loc[filtered_val_idx]
            y_pred_filt = reg_model.predict(X_val_filt)
            
            mae_val = mean_absolute_error(y_val_filt, y_pred_filt)
            rmse_val = np.sqrt(mean_squared_error(y_val_filt, y_pred_filt))
            r2_val = r2_score(y_val_filt, y_pred_filt)
            
            st.info("💡 As métricas abaixo são calculadas na amostra de validação restrita aos filtros selecionados.")
        else:
            # Fallback to entire validation set if filters are too narrow
            mae_val = mean_absolute_error(y_val, y_pred_val)
            rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
            r2_val = r2_score(y_val, y_pred_val)
            y_val_filt = y_val
            y_pred_filt = y_pred_val
            st.warning("Amostra filtrada de validação muito pequena. Exibindo métricas globais.")
            
        st.metric("R² Score", f"{r2_val*100:.2f}%")
        st.metric("MAE (Erro Absoluto Médio)", f"${mae_val:,.2f}")
        st.metric("RMSE (Raiz do Erro Quadrático Médio)", f"${rmse_val:,.2f}")
        
        # Prediction vs Actual Scatter plot
        fig_scatter = px.scatter(
            x=y_val_filt,
            y=y_pred_filt,
            labels={'x': 'Preço Real ($)', 'y': 'Preço Previsto ($)'},
            opacity=0.7,
            color_discrete_sequence=['#2563eb']
        )
        min_v = min(y_val_filt.min(), y_pred_filt.min())
        max_v = max(y_val_filt.max(), y_pred_filt.max())
        fig_scatter.add_shape(
            type='line',
            x0=min_v, y0=min_v, x1=max_v, y1=max_v,
            line=dict(color='#ef4444', dash='dash', width=2)
        )
        fig_scatter.update_layout(
            title="Preço Real vs Preço Previsto",
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="section-card" style="height:100%;">', unsafe_allow_html=True)
        st.write("<div class='section-title'>Modelagem de Classificação (Preço Acima da Mediana)</div>", unsafe_allow_html=True)
        
        mediana_preco = results['mediana_preco']
        st.write(f"**Classe 1 (Alvo):** Imóveis com Preço de Venda superior à Mediana Global (${mediana_preco:,.2f})")
        st.write(f"**Classe 0 (Alvo):** Imóveis com Preço de Venda inferior ou igual à Mediana Global")
        
        # Calculate classification metrics on the filtered subset of classification validation set
        Xc_val = results['Xc_val']
        yc_val = results['yc_val']
        yc_pred_val = results['yc_pred_val']
        clf_model = results['clf_model']
        
        filtered_val_idx_c = df_filtered.index.intersection(Xc_val.index)
        
        if len(filtered_val_idx_c) > 2:
            yc_val_filt = yc_val.loc[filtered_val_idx_c]
            Xc_val_filt = Xc_val.loc[filtered_val_idx_c]
            yc_pred_filt = clf_model.predict(Xc_val_filt)
            
            acc = accuracy_score(yc_val_filt, yc_pred_filt)
            cm = confusion_matrix(yc_val_filt, yc_pred_filt, labels=[0, 1])
            rep = classification_report(yc_val_filt, yc_pred_filt, labels=[0, 1], output_dict=True, zero_division=0)
        else:
            acc = accuracy_score(yc_val, yc_pred_val)
            cm = confusion_matrix(yc_val, yc_pred_val, labels=[0, 1])
            rep = classification_report(yc_val, yc_pred_val, labels=[0, 1], output_dict=True, zero_division=0)
            st.warning("Amostra filtrada de validação muito pequena. Exibindo métricas globais.")
            
        st.metric("Acurácia de Classificação", f"{acc*100:.2f}%")
        
        st.write("**Relatório de Classificação Segmentado:**")
        df_rep = pd.DataFrame(rep).transpose().round(4)
        st.dataframe(df_rep)
        
        # Confusion matrix visual
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=['Classe 0 (<= Mediana)', 'Classe 1 (> Mediana)'],
            y=['Classe 0 (<= Mediana)', 'Classe 1 (> Mediana)'],
            color_continuous_scale='Blues',
            labels=dict(x="Classe Prevista", y="Classe Real")
        )
        fig_cm.update_layout(
            title="Matriz de Confusão",
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------- ABA 3: APRENDIZADO NÃO SUPERVISIONADO -----------------
with tab3:
    st.write("### Aprendizado Não Supervisionada & Visão Espacial")
    
    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.write("<div class='section-title'>Filtro de Outliers (Isolation Forest)</div>", unsafe_allow_html=True)
        
        # Outlier counts on full dataset
        num_outliers_total = (df_processed['Outlier'] == -1).sum()
        st.write(f"Total de Outliers no dataset global: **{num_outliers_total} imóveis** (5% de contaminação)")
        
        # Option to show/hide outliers
        outlier_filter = st.radio(
            "Selecione o filtro espacial de Outliers:",
            ["Mostrar Todos os Imóveis", "Mostrar Apenas Outliers", "Filtrar (Excluir Outliers)"]
        )
        
        # Apply outlier filter on the filtered dataframe
        if outlier_filter == "Mostrar Apenas Outliers":
            df_unsup_display = df_filtered[df_filtered['Outlier'] == -1]
        elif outlier_filter == "Filtrar (Excluir Outliers)":
            df_unsup_display = df_filtered[df_filtered['Outlier'] == 1]
        else:
            df_unsup_display = df_filtered
            
        st.write(f"Imóveis exibidos no espaço PCA: **{df_unsup_display.shape[0]}**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-card">
            <div class="section-title">Perfis das Classes (K-Means)</div>
            <p>O algoritmo K-Means agrupou as propriedades em 3 perfis distintos baseados em seu espaço de features reduzido:</p>
            <ul>
                <li>🟢 <b>Populares:</b> Casas com menor área, menor qualidade geral e preços acessíveis.</li>
                <li>🔵 <b>Médios:</b> Perfil intermediário com área média e qualidade residencial equilibrada.</li>
                <li>🔴 <b>Luxo:</b> Residências de alto padrão, grande área útil e maior valor de mercado.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_r:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.write("<div class='section-title'>Espaço PCA 2D colorida por Perfil (K-Means)</div>", unsafe_allow_html=True)
        
        # Add labels to outliers for mapping
        df_unsup_display_plot = df_unsup_display.copy()
        df_unsup_display_plot['Outlier_Label'] = df_unsup_display_plot['Outlier'].map({1: 'Normal', -1: 'Outlier (Anomalia)'})
        
        # Color mapping for Profiles (Luxo, Médios, Populares)
        color_map = {
            "Populares": "#22c55e", # Green
            "Médios": "#3b82f6",    # Blue
            "Luxo": "#ef4444"       # Red
        }
        
        fig_pca_scatter = px.scatter(
            df_unsup_display_plot,
            x='PCA1',
            y='PCA2',
            color='Perfil',
            symbol='Outlier_Label',
            symbol_sequence=['circle', 'x'],
            color_discrete_map=color_map,
            hover_data={'SalePrice': ':$,.2f', 'OverallQual': True, 'Neighborhood': False},
            labels={'Perfil': 'Perfil do Imóvel', 'Outlier_Label': 'Status Espacial'}
        )
        
        fig_pca_scatter.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9', title="Componente Principal 1 (PCA1)"),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9', title="Componente Principal 2 (PCA2)"),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_pca_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)