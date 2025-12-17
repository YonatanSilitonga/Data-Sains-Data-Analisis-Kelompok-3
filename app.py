"""
RDKK Streamlit Dashboard - Layout Improvements
Perbaikan pada toggle placement dan spacing issues
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import yaml
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

from src.standards import StandardsManager, get_standard_manager
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import feature_engineering_pipeline
from src.clustering import run_clustering_pipeline
from src.recommendation import generate_recommendations

from src.anomaly_explain import get_anomaly_explanation, get_anomaly_explanation, get_anomaly_comparison, calculate_median_and_std
from src.utils import *
from src.standard_comparison import (
    calculate_standard_comparison,
    generate_explanation_text,
    get_educational_box,
    classify_overuse_level
)

# Load config
@st.cache_resource
def load_config():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config()

def get_all_commodities_from_config():
    """Ambil semua komoditas dari config.yaml standar_pupuk"""
    standards = config.get('standar_pupuk', {})
    if not standards:
        return []
    # Filter out 'enabled' key
    commodities = [k for k in standards.keys() if k != 'enabled']
    return sorted(commodities)

# Initialize session state
if 'standards_manager' not in st.session_state:
    st.session_state.standards_manager = get_standard_manager()
if 'standards_enabled' not in st.session_state:
    st.session_state.standards_enabled = config.get('standar_pupuk', {}).get('enabled', True)

standards_manager = st.session_state.standards_manager

# IMPROVED CSS - Fixed spacing and better layout
st.markdown("""
<style>
    /* Remove default Streamlit padding */
    .main .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 1.5rem;
    }
    
    /* Toggle container - Better positioning */
    .toggle-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1.5rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }
    
    .status-active {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .status-inactive {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1976d2;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1976d2;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .danger-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Full width for content */
    .element-container {
        width: 100% !important;
    }
    
    /* Better dataframe display */
    .stDataFrame {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_data():
    """Load processed data"""
    data_path = Path("output/dataset_final.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    return None

@st.cache_resource
def load_models():
    """Load trained models"""
    models_dir = Path("models")
    models = {}
    
    scaler_path = models_dir / "scaler.joblib"
    kmeans_path = models_dir / "kmeans.joblib"
    features_path = models_dir / "features.json"
    
    if scaler_path.exists() and kmeans_path.exists() and features_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            kmeans = joblib.load(kmeans_path)
            with open(features_path, 'r') as f:
                features_info = json.load(f)
            
            models['clustering'] = {
                'scaler': scaler,
                'model': kmeans,
                'feature_cols': features_info['feature_cols'],
                'n_clusters': features_info['n_clusters'],
                'characteristics': features_info.get('characteristics', {})
            }
        except Exception as e:
            st.error(f"Error loading models: {e}")
    
    return models

# Load data and models
df = load_data()
models = load_models()

# IMPROVED HEADER LAYOUT
st.markdown('<div class="main-header">🌾 RDKK - Sistem Analisis Pupuk Subsidi</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dashboard Interaktif untuk Deteksi Over/Under Use Pupuk dengan Machine Learning</div>', unsafe_allow_html=True)

# REST OF YOUR CODE CONTINUES HERE...
# The sidebar with toggle at the top

with st.sidebar:
    st.title("Menu")
    st.markdown("---")       
    
    standards_enabled = st.toggle(
        "Aktifkan Standar Pupuk", 
        value=st.session_state.standards_enabled,
        help="Toggle ON: Analisis menggunakan standar pupuk per komoditas\nToggle OFF: Analisis berbasis data aktual saja",
        key="sidebar_toggle"
    )
    
    if standards_enabled != st.session_state.standards_enabled:
        st.session_state.standards_enabled = standards_enabled
        st.rerun()
    
    # Status indicator kompak
    if standards_enabled:
        st.success("**Mode:** Standar Aktif", icon="✅")
    else:
        st.info("**Mode:** Data Aktual", icon="ℹ️")
    
    st.markdown("---")
    
    # Navigation menu
    page = st.radio(
        "Pilih Halaman:",
        [
            "Dashboard Utama",
            "Data Explorer", 
            "Prediksi Data Baru",
            "Clustering & Pola",
            "Rekomendasi",
            "Tentang Model",
            "Kelola Standar"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Info Sistem")
    
    if df is not None:
        st.metric("Total Petani", f"{len(df):,}")
        
        if standards_enabled and 'Final_Status' in df.columns:
            underuse = (df['Final_Status'] == 'Underuse').sum()
            overuse = (df['Final_Status'] == 'Overuse').sum()
            normal = (df['Final_Status'] == 'Normal').sum()
            
            st.metric("🟢 Normal", f"{normal:,}", f"{normal/len(df)*100:.1f}%")
            st.metric("🟡 Underuse", f"{underuse:,}", f"{underuse/len(df)*100:.1f}%")
            st.metric("🔴 Overuse", f"{overuse:,}", f"{overuse/len(df)*100:.1f}%")
        
        if 'Cluster_ID' in df.columns:
            st.metric("Clusters", df['Cluster_ID'].nunique())
    else:
        st.warning("⚠️ Data belum tersedia\n\nJalankan pipeline:\n`python main.py`")
    
    st.markdown("---")
    st.info("🔧 Version: 4.1.0\n📅 Updated: Dec 2025")

# PAGE CONTENT CONTINUES HERE...
# (All your existing page logic remains the same)

if page == "Dashboard Utama":
    
    if df is None:
        st.error("❌ Data tidak tersedia. Jalankan `python main.py` terlebih dahulu untuk memproses data.")
        st.code("python main.py", language="bash")
        st.stop()
    
    st.markdown('<div class="section-header">Ringkasan Cepat</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Petani", f"{len(df):,}")
    
    with col2:
        if standards_enabled and 'Final_Status' in df.columns:
            underuse = (df['Final_Status'] == 'Underuse').sum()
            st.metric("Underuse", underuse, 
                     delta=f"-{underuse/len(df)*100:.1f}%", 
                     delta_color="inverse")
        else:
            st.metric("Underuse", "N/A", help="Aktifkan standar pupuk")
    
    with col3:
        if standards_enabled and 'Final_Status' in df.columns:
            overuse = (df['Final_Status'] == 'Overuse').sum()
            st.metric("Overuse", overuse,
                     delta=f"+{overuse/len(df)*100:.1f}%",
                     delta_color="inverse")
        else:
            st.metric("Overuse", "N/A", help="Aktifkan standar pupuk")
    
    with col4:
        if 'Cluster_ID' in df.columns:
            clusters = df['Cluster_ID'].nunique()
            st.metric("Clusters", clusters)
        else:
            st.metric("Clusters", "N/A")
    
    st.markdown("---")
    
    if standards_enabled:
        st.markdown('<div class="section-header">Distribusi Status Penggunaan Pupuk</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            status_counts = df['Final_Status'].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title='Distribusi Status (Semua Petani)',
                color=status_counts.index,
                color_discrete_map={'Normal': '#4caf50', 'Underuse': '#ff9800', 'Overuse': '#f44336'},
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart per komoditas
            if 'Komoditas' in df.columns:
                status_by_commodity = df.groupby(['Komoditas', 'Final_Status']).size().reset_index(name='Jumlah')
                fig = px.bar(
                    status_by_commodity,
                    x='Komoditas',
                    y='Jumlah',
                    color='Final_Status',
                    title='Status per Komoditas',
                    color_discrete_map={'Normal': '#4caf50', 'Underuse': '#ff9800', 'Overuse': '#f44336'},
                    barmode='group'
                )
                fig.update_layout(xaxis_title="Komoditas", yaxis_title="Jumlah Petani")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Aktifkan standar pupuk untuk melihat distribusi status underuse/overuse")
    
    if standards_enabled:
        st.markdown('<div class="section-header">Perbandingan: Pupuk Aktual vs Batas Maksimal Standar</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>📌 Penting:</strong> Standar pupuk adalah <strong>batas atas hak/alokasi maksimal</strong>, bukan target ideal. 
        Grafik menunjukkan batas maksimal sebagai garis pembatas. Petani boleh menggunakan di bawah batas ini.
        </div>
        """, unsafe_allow_html=True)
        
        all_standards = standards_manager.get_all_standards()
        
        if all_standards and all(col in df.columns for col in ['Komoditas', 'Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha']):
            komoditas_list = []
            actual_urea = []
            actual_npk = []
            actual_organik = []
            std_urea_max = []
            std_npk_max = []
            std_organik_max = []
            
            for komoditas in all_standards.keys():
                komoditas_data = df[df['Komoditas'] == komoditas]
                if len(komoditas_data) > 0:
                    komoditas_list.append(komoditas)
                    actual_urea.append(komoditas_data['Urea_per_ha'].mean())
                    actual_npk.append(komoditas_data['NPK_per_ha'].mean())
                    actual_organik.append(komoditas_data['Organik_per_ha'].mean())
                    
                    std = standards_manager.get_standard(komoditas)
                    std_urea_max.append(std['Urea']['max'])
                    std_npk_max.append(std['NPK']['max'])
                    std_organik_max.append(std['Organik']['max'])
            
            if komoditas_list:
                tab1, tab2, tab3 = st.tabs(["🌾 Urea", "🌱 NPK", "🍂 Organik"])
                
                with tab1:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=komoditas_list, 
                        y=actual_urea, 
                        name='Rata-rata Aktual', 
                        marker_color='#66b5f6'
                    ))
                    fig.add_trace(go.Scatter(
                        x=komoditas_list, 
                        y=std_urea_max, 
                        name='Batas Maksimal', 
                        mode='lines+markers',
                        line=dict(color='#d32f2f', width=3, dash='dash'),
                        marker=dict(size=8, color='#d32f2f')
                    ))
                    fig.update_layout(
                        title="Urea: Penggunaan Aktual vs Batas Maksimal (kg/ha)", 
                        height=400,
                        xaxis_title="Komoditas",
                        yaxis_title="kg/ha",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=komoditas_list, 
                        y=actual_npk, 
                        name='Rata-rata Aktual', 
                        marker_color='#81c784'
                    ))
                    fig.add_trace(go.Scatter(
                        x=komoditas_list, 
                        y=std_npk_max, 
                        name='Batas Maksimal', 
                        mode='lines+markers',
                        line=dict(color='#d32f2f', width=3, dash='dash'),
                        marker=dict(size=8, color='#d32f2f')
                    ))
                    fig.update_layout(
                        title="NPK: Penggunaan Aktual vs Batas Maksimal (kg/ha)", 
                        height=400,
                        xaxis_title="Komoditas",
                        yaxis_title="kg/ha",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=komoditas_list, 
                        y=actual_organik, 
                        name='Rata-rata Aktual', 
                        marker_color='#ffb74d'
                    ))
                    fig.add_trace(go.Scatter(
                        x=komoditas_list, 
                        y=std_organik_max, 
                        name='Batas Maksimal', 
                        mode='lines+markers',
                        line=dict(color='#d32f2f', width=3, dash='dash'),
                        marker=dict(size=8, color='#d32f2f')
                    ))
                    fig.update_layout(
                        title="Organik: Penggunaan Aktual vs Batas Maksimal (kg/ha)", 
                        height=400,
                        xaxis_title="Komoditas",
                        yaxis_title="kg/ha",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-header">Analisis Luas Lahan & Intensitas Pupuk</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Luas_Tanah_ha' in df.columns:
            fig = px.histogram(
                df, 
                x='Luas_Tanah_ha', 
                nbins=30, 
                title='Distribusi Luas Lahan (ha)',
                color_discrete_sequence=['#2196f3']
            )
            fig.update_layout(xaxis_title="Luas Lahan (ha)", yaxis_title="Jumlah Petani")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Total_per_ha' in df.columns:
            fig = px.histogram(
                df, 
                x='Total_per_ha', 
                nbins=30, 
                title='Distribusi Intensitas Pupuk Total (kg/ha)',
                color_discrete_sequence=['#4caf50']
            )
            fig.update_layout(xaxis_title="Total Pupuk (kg/ha)", yaxis_title="Jumlah Petani")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("""
            ⚠️ **Visualisasi Total Pupuk per ha tidak tersedia**
            
            Kolom `Total_per_ha` tidak ditemukan dalam data. 
            
            **Penyebab:**
            - Feature engineering belum menghasilkan kolom ini
            - Pipeline belum dijalankan dengan versi terbaru
            
            **Solusi:**
            Jalankan ulang pipeline:
            \`\`\`
            python main.py
            \`\`\`
            """)

# ==========================================
# PAGE 2: DATA EXPLORER (ENHANCED WITH VALIDATION)
# ==========================================
elif page == "Data Explorer":
    st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)
    
    if df is None:
        st.error("❌ Data tidak tersedia")
        st.stop()
    
    st.markdown("""
    <div class="info-box">
    <strong>Pahami Dulu:</strong><br>
    <ul style="margin: 10px 0; padding-left: 20px;">
        <li><strong>NORMAL</strong> = Mengikuti pola mayoritas petani dalam penggunaan total pupuk per jenis (bukan berarti benar/salah)</li>
        <li><strong>ANOMALI</strong> = Berbeda dari pola umum dalam penggunaan total pupuk per jenis (bukan pelanggaran)</li>
        <li>Fokus analisis: <strong>Total Urea, NPK, Organik per hektar</strong> (tidak berdasarkan MT)</li>
        <li>Anomali bisa lebih baik atau lebih buruk dari normal - yang penting berbeda dari mayoritas</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    all_commodities_config = get_all_commodities_from_config()
    if 'Komoditas' in df.columns:
        data_commodities = set(df['Komoditas'].dropna().unique())
        config_commodities = set(all_commodities_config)
        
        invalid_commodities = data_commodities - config_commodities
        if invalid_commodities:
            st.warning(f"""
            **Perhatian:** Ada {len(invalid_commodities)} komoditas di data yang tidak ada di config:
            **{', '.join(sorted(invalid_commodities))}**
            
            Komoditas valid hanya: **{', '.join(sorted(config_commodities))}**
            
            Data dengan komoditas tidak valid mungkin tidak ditampilkan dengan benar.
            """)
    
    # Filter controls
    st.markdown("### Filter Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'Komoditas' in df.columns:
            valid_commodities_in_data = [k for k in df['Komoditas'].dropna().unique() if k in all_commodities_config]
            komoditas_options = ['Semua'] + sorted(valid_commodities_in_data)
            selected_komoditas = st.selectbox("Komoditas", komoditas_options)
        else:
            st.warning("Kolom Komoditas tidak tersedia")
            selected_komoditas = 'Semua'
    
    with col2:
        if 'Anomaly_Label' in df.columns:
            anomaly_options = ['Semua'] + sorted(df['Anomaly_Label'].unique().tolist())
            selected_anomaly = st.selectbox("Label Anomali", anomaly_options)
        else:
            selected_anomaly = 'Semua'
    
    with col3:
        if standards_enabled and 'Final_Status' in df.columns:
            status_options = ['Semua'] + sorted(df['Final_Status'].unique().tolist())
            selected_status = st.selectbox("Status Standar", status_options)
        else:
            selected_status = 'Semua'
    
    # Apply filters
    df_filtered = df.copy()
    
    if selected_komoditas != 'Semua' and 'Komoditas' in df.columns:
        df_filtered = df_filtered[df_filtered['Komoditas'] == selected_komoditas]
    
    if selected_anomaly != 'Semua' and 'Anomaly_Label' in df.columns:
        df_filtered = df_filtered[df_filtered['Anomaly_Label'] == selected_anomaly]
    
    if selected_status != 'Semua' and 'Final_Status' in df.columns:
        df_filtered = df_filtered[df_filtered['Final_Status'] == selected_status]
    
    if len(df_filtered) == 0:
        st.warning("""
        ⚠️ **Tidak ada data untuk kombinasi filter ini**
        
        Silakan ubah filter atau pilih 'Semua' untuk melihat data.
        """)
        st.stop()
    
    st.markdown(f"**Total data setelah filter:** {len(df_filtered):,} petani")
    
    st.markdown("---")
    
    st.markdown("### Tabel Data")
    
    st.markdown("""
    <div class="info-box">
    <strong>Tabel Data Petani:</strong> Lihat detail lengkap setiap petani sebelum melihat visualisasi.
    Tabel ini dapat diurutkan dengan klik header kolom.
    </div>
    """, unsafe_allow_html=True)
    
    # Select relevant columns for display
    table_columns = ['ID_Petani', 'Komoditas', 'Desa', 'Luas_Tanah_ha']
    
    # Add fertilizer columns
    if 'Urea_per_ha' in df_filtered.columns:
        table_columns.append('Urea_per_ha')
    if 'NPK_per_ha' in df_filtered.columns:
        table_columns.append('NPK_per_ha')
    if 'Organik_per_ha' in df_filtered.columns:
        table_columns.append('Organik_per_ha')
    if 'Total_per_ha' in df_filtered.columns:
        table_columns.append('Total_per_ha')
    
    # Add status columns
    if 'Anomaly_Label' in df_filtered.columns:
        table_columns.append('Anomaly_Label')
    if standards_enabled and 'Final_Status' in df_filtered.columns:
        table_columns.append('Final_Status')
    
    # Filter columns that exist
    table_columns = [col for col in table_columns if col in df_filtered.columns]
    
    # Display table
    df_table = df_filtered[table_columns].copy()
    
    # Format numeric columns for better readability
    numeric_cols = ['Luas_Tanah_ha', 'Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha', 'Total_per_ha']
    for col in numeric_cols:
        if col in df_table.columns:
            df_table[col] = df_table[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # Download button
    csv = df_table.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Data (CSV)",
        data=csv,
        file_name=f"data_explorer_{selected_komoditas}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
    
    st.markdown("---")
    st.markdown("### Visualisasi Data")
    
    st.markdown("#### 1️⃣ Distribusi Total Pupuk per Hektar per Komoditas")
    
    if 'Komoditas' in df_filtered.columns and 'Total_per_ha' in df_filtered.columns:
        # Validasi data tidak kosong
        if df_filtered['Total_per_ha'].notna().sum() > 0:
            fig = px.box(
                df_filtered,
                x='Komoditas',
                y='Total_per_ha',
                color='Komoditas',
                title='Distribusi Total Pupuk per Hektar (Semua Jenis)',
                labels={'Total_per_ha': 'Total Pupuk (kg/ha)', 'Komoditas': 'Komoditas'}
            )
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
            <strong>Cara Baca:</strong><br>
            Grafik ini menunjukkan distribusi <strong>TOTAL pupuk per hektar</strong> (Urea + NPK + Organik) untuk setiap komoditas.
            Box menunjukkan rentang nilai mayoritas, titik di luar box = outlier/anomali.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Data Total_per_ha kosong untuk filter ini")
    else:
        missing_cols = []
        if 'Komoditas' not in df_filtered.columns:
            missing_cols.append('Komoditas')
        if 'Total_per_ha' not in df_filtered.columns:
            missing_cols.append('Total_per_ha')
        
        st.error(f"""
        ❌ **Visualisasi tidak dapat ditampilkan**
        
        Kolom yang hilang: {', '.join(missing_cols)}
        
        <strong>Solusi:</strong>
        1. Pastikan data memiliki kolom Komoditas
        2. Jalankan ulang pipeline untuk menghasilkan Total_per_ha:
        \`\`\`
        python main.py
        \`\`\`
        """)
    
    st.markdown("---")
    
    if standards_enabled and 'Final_Status' in df_filtered.columns and 'Komoditas' in df_filtered.columns:
        st.markdown("#### 2️⃣ Status Penggunaan Pupuk per Komoditas (Standar Aktif)")
        
        if len(df_filtered) > 0:
            status_by_commodity = df_filtered.groupby(['Komoditas', 'Final_Status']).size().reset_index(name='Jumlah')
            
            if len(status_by_commodity) > 0:
                fig = px.bar(
                    status_by_commodity,
                    x='Komoditas',
                    y='Jumlah',
                    color='Final_Status',
                    title='Status Penggunaan Pupuk per Komoditas',
                    color_discrete_map={'Normal': '#4caf50', 'Underuse': '#ff9800', 'Overuse': '#f44336'},
                    barmode='stack'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ Tidak ada data status untuk filter ini")
        else:
            st.warning("⚠️ Tidak ada data untuk ditampilkan")
    
    st.markdown("---")
    
    st.markdown('<div class="section-header">DETAIL LENGKAP PETANI INDIVIDUAL</div>', unsafe_allow_html=True)
    
    selected_id = st.selectbox(
        "Pilih ID Petani untuk Analisis Detail:",
        ['-- Pilih --'] + df_filtered['ID_Petani'].tolist()
    )
    
    if selected_id != '-- Pilih --':
        petani_data = df_filtered[df_filtered['ID_Petani'] == selected_id].iloc[0]
        komoditas_petani = petani_data.get('Komoditas', 'Unknown')
        
        df_same_commodity = df_filtered[df_filtered['Komoditas'] == komoditas_petani]
        
        if len(df_same_commodity) < 2:
            st.warning(f"⚠️ Tidak cukup data untuk perbandingan komoditas {komoditas_petani}")
        else:
            st.markdown(f"### 📋 Detail Petani: {selected_id} ({komoditas_petani})")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📍 Informasi Dasar**")
                st.write(f"Desa: {petani_data.get('Desa', 'N/A')}")
                st.write(f"Kelompok: {petani_data.get('Kelompok_Tani', 'N/A')}")
                st.write(f"Komoditas: {komoditas_petani}")
                st.write(f"Luas Lahan: {petani_data.get('Luas_Tanah_ha', 0):.2f} ha")
            
            with col2:
                st.markdown("**🌾 Total Pupuk Digunakan**")
                if 'Total_Urea' in petani_data.index:
                    st.write(f"Total Urea: {petani_data.get('Total_Urea', 0):.1f} kg")
                if 'Total_NPK' in petani_data.index:
                    st.write(f"Total NPK: {petani_data.get('Total_NPK', 0):.1f} kg")
                if 'Total_Organik' in petani_data.index:
                    st.write(f"Total Organik: {petani_data.get('Total_Organik', 0):.1f} kg")
            
            with col3:
                st.markdown("**Pupuk per Hektar**")
                if 'Urea_per_ha' in petani_data.index:
                    st.write(f"Urea: {petani_data.get('Urea_per_ha', 0):.1f} kg/ha")
                if 'NPK_per_ha' in petani_data.index:
                    st.write(f"NPK: {petani_data.get('NPK_per_ha', 0):.1f} kg/ha")
                if 'Organik_per_ha' in petani_data.index:
                    st.write(f"Organik: {petani_data.get('Organik_per_ha', 0):.1f} kg/ha")
            
            st.markdown("---")
            
            st.markdown("#### VISUALISASI 1: Proporsi Penggunaan Pupuk")
            st.markdown("*Membandingkan komposisi pupuk dengan median komoditas*")
            
            total_urea = petani_data.get('Total_Urea', 0)
            total_npk = petani_data.get('Total_NPK', 0)
            total_organik = petani_data.get('Total_Organik', 0)
            total_all = total_urea + total_npk + total_organik
            
            if total_all > 0 and pd.notna(total_all):
                prop_urea = (total_urea / total_all) * 100
                prop_npk = (total_npk / total_all) * 100
                prop_organik = (total_organik / total_all) * 100
                
                df_commodity_valid = df_same_commodity.copy()
                
                # Check if Total columns exist
                if all(col in df_commodity_valid.columns for col in ['Total_Urea', 'Total_NPK', 'Total_Organik']):
                    df_commodity_valid['total_pupuk'] = (
                        df_commodity_valid['Total_Urea'] + 
                        df_commodity_valid['Total_NPK'] + 
                        df_commodity_valid['Total_Organik']
                    )
                    
                    df_commodity_valid = df_commodity_valid[df_commodity_valid['total_pupuk'] > 0]
                    
                    if len(df_commodity_valid) > 0:
                        df_commodity_valid['Prop_Urea'] = (df_commodity_valid['Total_Urea'] / 
                                                           df_commodity_valid['total_pupuk']) * 100
                        df_commodity_valid['Prop_NPK'] = (df_commodity_valid['Total_NPK'] / 
                                                          df_commodity_valid['total_pupuk']) * 100
                        df_commodity_valid['Prop_Organik'] = (df_commodity_valid['Total_Organik'] / 
                                                              df_commodity_valid['total_pupuk']) * 100
                        
                        median_prop_urea = df_commodity_valid['Prop_Urea'].median()
                        median_prop_npk = df_commodity_valid['Prop_NPK'].median()
                        median_prop_organik = df_commodity_valid['Prop_Organik'].median()
                        
                        categories = ['Urea', 'NPK', 'Organik']
                        petani_props = [prop_urea, prop_npk, prop_organik]
                        median_props = [median_prop_urea, median_prop_npk, median_prop_organik]
                        
                        fig_prop = go.Figure()
                        
                        fig_prop.add_trace(go.Bar(
                            x=categories,
                            y=median_props,
                            name=f'Median {komoditas_petani}',
                            marker_color='#42A5F5',
                            text=[f'{v:.1f}%' for v in median_props],
                            textposition='auto'
                        ))
                        
                        fig_prop.add_trace(go.Bar(
                            x=categories,
                            y=petani_props,
                            name=f'Petani {selected_id}',
                            marker_color='#FF7043',
                            text=[f'{v:.1f}%' for v in petani_props],
                            textposition='auto'
                        ))
                        
                        fig_prop.update_layout(
                            title=f'Proporsi Penggunaan Pupuk (%)',
                            barmode='group',
                            height=400,
                            yaxis_title='Proporsi (%)',
                            xaxis_title='Jenis Pupuk',
                            yaxis=dict(range=[0, 100])
                        )
                        
                        st.plotly_chart(fig_prop, use_container_width=True)
                        
                        st.markdown("""
                        <div class="info-box">
                        <strong>Interpretasi:</strong> Grafik ini menunjukkan komposisi penggunaan pupuk petani 
                        dibandingkan dengan pola umum (median). Jika distribusi sangat berbeda dari median, 
                        maka akan terdeteksi sebagai anomali.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("Tidak cukup data untuk menghitung proporsi median komoditas.")
                else:
                    st.warning("Kolom Total_Urea, Total_NPK, atau Total_Organik tidak tersedia dalam dataset.")
            else:
                st.warning("Total pupuk petani adalah 0, tidak dapat menghitung proporsi.")
            
            # Show anomaly status
            if 'Anomaly_Label' in petani_data.index:
                anomaly_label = petani_data.get('Anomaly_Label', 'Normal')
                anomaly_score = petani_data.get('Anomaly_Score', 0.0)
                
                if anomaly_label == 'Anomali':
                    st.error(f"⚠️ Status: **ANOMALI** (Skor: {anomaly_score:.3f})")
                else:
                    st.success(f"✅ Status: **NORMAL** (Skor: {anomaly_score:.3f})")
                
                try:
                    from src.anomaly_explain import get_anomaly_explanation
                    explanation = get_anomaly_explanation(selected_id, df_filtered, standards_manager)
                    
                    st.markdown("#### Penjelasan Anomali")
                    st.markdown(explanation['explanation_text'])
                except Exception as e:
                    st.warning(f"Tidak dapat memuat penjelasan detail: {e}")
            
            st.markdown("---")
            
            st.markdown("#### VISUALISASI 2: Total Pupuk (kg) - Petani vs Median")
            st.markdown("*Membandingkan total penggunaan pupuk dengan pola mayoritas petani*")
            
            comparison_data = []
            pupuk_types_viz2 = []
            petani_values_viz2 = []
            median_values_viz2 = []
            
            for pupuk in ['Urea', 'NPK', 'Organik']:
                col_total = f'Total_{pupuk}'
                if col_total in petani_data.index and col_total in df_same_commodity.columns:
                    petani_val = petani_data[col_total]
                    median_val = df_same_commodity[col_total].median()
                    
                    # Validate values
                    if pd.notna(petani_val):
                        if pd.isna(median_val):
                            median_val = 0.0
                        
                        pupuk_types_viz2.append(f'{pupuk}')
                        petani_values_viz2.append(petani_val)
                        median_values_viz2.append(median_val)
                        
                        if median_val > 0:
                            diff_pct = ((petani_val - median_val) / median_val * 100)
                        else:
                            # If median is 0 but petani value exists, it's infinite difference
                            if petani_val > 0:
                                diff_pct = 999.9  # Indicate very large positive difference
                            else:
                                diff_pct = 0.0
                        
                        comparison_data.append({
                            'Jenis Pupuk': pupuk,
                            'Petani': f"{petani_val:.1f} kg",
                            'Median': f"{median_val:.1f} kg",
                            'Selisih': f"{diff_pct:+.1f}%" if diff_pct < 999 else ">>100%"
                        })
            
            if comparison_data:
                # Show table
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Show chart
                fig_compare = go.Figure()
                
                fig_compare.add_trace(go.Bar(
                    x=pupuk_types_viz2,
                    y=petani_values_viz2,
                    name=f'Petani {selected_id}',
                    marker_color='#FF5722',
                    text=[f'{v:.1f}' for v in petani_values_viz2],
                    textposition='auto'
                ))
                
                fig_compare.add_trace(go.Bar(
                    x=pupuk_types_viz2,
                    y=median_values_viz2,
                    name=f'Median {komoditas_petani}',
                    marker_color='#4CAF50',
                    text=[f'{v:.1f}' for v in median_values_viz2],
                    textposition='auto'
                ))
                
                fig_compare.update_layout(
                    title=f"Perbandingan Total Pupuk: Petani {selected_id} vs Median {komoditas_petani}",
                    yaxis_title="Total Pupuk (kg)",
                    barmode='group',
                    height=450
                )
                
                st.plotly_chart(fig_compare, use_container_width=True)
                
                interpretations = []
                for i, data in enumerate(comparison_data):
                    selisih_str = data['Selisih'].replace('%', '').replace('+', '').strip()
                    try:
                        pupuk_name = data['Jenis Pupuk']
                        
                        # Handle special case for very large differences
                        if selisih_str == ">>100":
                            interpretations.append(f"🔴 {pupuk_name}: Penggunaan JAUH LEBIH TINGGI dari median (median = 0, petani menggunakan pupuk ini)")
                        else:
                            selisih_pct = float(selisih_str)
                            
                            if abs(selisih_pct) < 10:
                                interpretations.append(f"✅ {pupuk_name}: Penggunaan sesuai pola normal (selisih {selisih_pct:+.1f}%)")
                            elif selisih_pct > 30:
                                interpretations.append(f"🔴 {pupuk_name}: Penggunaan JAUH LEBIH TINGGI dari median (+{selisih_pct:.1f}%)")
                            elif selisih_pct < -30:
                                interpretations.append(f"🟡 {pupuk_name}: Penggunaan JAUH LEBIH RENDAH dari median ({selisih_pct:.1f}%)")
                            elif selisih_pct > 0:
                                interpretations.append(f"🟠 {pupuk_name}: Sedikit lebih tinggi dari median (+{selisih_pct:.1f}%)")
                            else:
                                interpretations.append(f"🟠 {pupuk_name}: Sedikit lebih rendah dari median ({selisih_pct:.1f}%)")
                    except Exception as e:
                        interpretations.append(f"⚠️ {pupuk_name}: Tidak dapat menghitung interpretasi")
                
                st.markdown("""
                <div class="info-box">
                <strong>Interpretasi Grafik:</strong><br>
                {}
                </div>
                """.format('<br>'.join(interpretations)), unsafe_allow_html=True)
            else:
                st.warning("⚠️ Data Total pupuk tidak tersedia untuk perbandingan.")

            st.markdown("---")
            
            if standards_enabled:
                st.markdown("#### VISUALISASI 3: Perbandingan dengan Standar Pupuk")
                st.markdown("*Membandingkan penggunaan aktual dengan batas maksimal standar*")
                
                std = standards_manager.get_standard(komoditas_petani)
                
                if std:
                    pupuk_types_std = []
                    petani_values_std = []
                    std_max_values = []
                    
                    for pupuk in ['Urea', 'NPK', 'Organik']:
                        col_per_ha = f'{pupuk}_per_ha'
                        if col_per_ha in petani_data.index and pupuk in std:
                            petani_val = petani_data[col_per_ha]
                            std_max = std[pupuk]['max']
                            
                            if pd.notna(petani_val):
                                pupuk_types_std.append(f'{pupuk}/ha')
                                petani_values_std.append(petani_val)
                                std_max_values.append(std_max)
                    
                    if pupuk_types_std:
                        fig_std = go.Figure()
                        
                        fig_std.add_trace(go.Bar(
                            x=pupuk_types_std,
                            y=petani_values_std,
                            name=f'Petani {selected_id}',
                            marker_color='#2196F3',
                            text=[f'{v:.1f}' for v in petani_values_std],
                            textposition='auto'
                        ))
                        
                        fig_std.add_trace(go.Scatter(
                            x=pupuk_types_std,
                            y=std_max_values,
                            name='Batas Maksimal',
                            mode='lines+markers',
                            line=dict(color='#F44336', width=3, dash='dash'),
                            marker=dict(size=10, color='#F44336'),
                            text=[f'{v:.1f}' for v in std_max_values],
                            textposition='top center'
                        ))
                        
                        fig_std.update_layout(
                            title=f"Penggunaan Aktual vs Batas Maksimal Standar",
                            yaxis_title="kg/ha",
                            height=450,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_std, use_container_width=True)
                        
                        st.markdown("""
                        <div class="info-box">
                        <strong>📌 Interpretasi:</strong> Garis merah putus-putus menunjukkan <strong>batas maksimal</strong> 
                        penggunaan pupuk berdasarkan standar. Jika bar biru melebihi garis merah, maka terjadi overuse.
                        <br><br>
                        <strong>Catatan:</strong> Standar adalah batas atas, bukan target. Penggunaan di bawah batas adalah normal.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show status for each fertilizer type
                        col1, col2, col3 = st.columns(3)
                        
                        for i, pupuk in enumerate(['Urea', 'NPK', 'Organik']):
                            col_per_ha = f'{pupuk}_per_ha'
                            status_col = f'Status_{pupuk}'
                            
                            if status_col in petani_data.index:
                                status = petani_data[status_col]
                                
                                with [col1, col2, col3][i]:
                                    if status == 'Overuse':
                                        st.markdown(f'<div class="danger-box"><strong>🌾 {pupuk}:</strong> Overuse 🚨<br>Melebihi batas standar</div>', unsafe_allow_html=True)
                                    elif status == 'Underuse':
                                        st.markdown(f'<div class="warning-box"><strong>🌾 {pupuk}:</strong> Underuse ⚠️<br>Di bawah standar minimal</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div class="success-box"><strong>🌾 {pupuk}:</strong> Normal ✅<br>Sesuai standar</div>', unsafe_allow_html=True)
                    else:
                        st.info("ℹ️ Data pupuk per hektar tidak tersedia untuk perbandingan standar")
                else:
                    st.warning(f"⚠️ Standar pupuk tidak ditemukan untuk komoditas {komoditas_petani}")
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "1. Distribusi Pupuk per Hektar", 
        "2. Lokasi & Intensitas", 
        "3. Persebaran per Desa", 
        "4. Boxplot Detail Pupuk"
    ])
    
    with viz_tab1:
        st.markdown("#### Distribusi Pupuk per Hektar (Seluruh Jenis)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Urea_per_ha' in df_filtered.columns:
                fig = px.box(
                    df_filtered,
                    x='Komoditas',
                    y='Urea_per_ha',
                    title='Urea (kg/ha)',
                    color='Komoditas',
                    points='outliers'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'NPK_per_ha' in df_filtered.columns:
                fig = px.box(
                    df_filtered,
                    x='Komoditas',
                    y='NPK_per_ha',
                    title='NPK (kg/ha)',
                    color='Komoditas',
                    points='outliers'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        if 'Organik_per_ha' in df_filtered.columns:
            fig = px.box(
                df_filtered,
                x='Komoditas',
                y='Organik_per_ha',
                title='Organik (kg/ha)',
                color='Komoditas',
                points='outliers'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab2:
        st.markdown(f"""
        <strong>Lokasi Geografis & Intensitas Penggunaan Pupuk:</strong><br>
        Visualisasi ini membantu melihat pola spasial penggunaan pupuk.
        Titik yang lebih besar menandakan total pupuk yang digunakan lebih banyak.
        """, unsafe_allow_html=True)
        
        if all(col in df_filtered.columns for col in ['Luas_Tanah_ha', 'Total_per_ha', 'Cluster_ID']):
            fig = px.scatter(
                df_filtered,
                x='Luas_Tanah_ha',
                y='Total_per_ha',
                color='Cluster_ID' if 'Cluster_ID' in df_filtered.columns else None,
                title='Intensitas Pupuk vs Luas Lahan per Cluster',
                labels={'Luas_Tanah_ha': 'Luas Lahan (ha)', 'Total_per_ha': 'Total Pupuk (kg)'},
                size='Total_Pupuk' if 'Total_Pupuk' in df_filtered.columns else 10,
                size_max=15,
                hover_data=['ID_Petani', 'Komoditas', 'Desa'] if all(col in df_filtered.columns for col in ['ID_Petani', 'Komoditas', 'Desa']) else None
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Data lokasi dan intensitas pupuk tidak lengkap untuk visualisasi ini.")
    
    with viz_tab3:
        if 'Desa' in df_filtered.columns:
            st.markdown(f"""
            <strong>Penjelasan:</strong> Distribusi petani dan penggunaan pupuk per desa. 
            Membantu identifikasi desa yang perlu perhatian khusus.
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                petani_by_desa = df_filtered['Desa'].value_counts().reset_index()
                petani_by_desa.columns = ['Desa', 'Jumlah']
                fig = px.pie(
                    petani_by_desa,
                    values='Jumlah',
                    names='Desa',
                    title='Distribusi Petani per Desa'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Total_per_ha' in df_filtered.columns:
                    pupuk_by_desa = df_filtered.groupby('Desa')['Total_per_ha'].mean().reset_index()
                    fig = px.bar(
                        pupuk_by_desa,
                        x='Desa',
                        y='Total_per_ha',
                        title='Rata-rata Intensitas Pupuk per Desa (kg/ha)',
                        color='Total_per_ha',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab4:
        st.markdown(f"""
        <strong>Penjelasan:</strong> Boxplot menunjukkan distribusi, median, dan outlier (nilai ekstrem) penggunaan pupuk.
        - <strong>Kotak</strong>: 50% data tengah
        - <strong>Garis tengah</strong>: Median (nilai tengah)
        - <strong>Titik di luar</strong>: Outlier (anomali potensial)
        """, unsafe_allow_html=True)
        
        if all(col in df_filtered.columns for col in ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha', 'Komoditas']):
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(
                    df_filtered,
                    x='Komoditas',
                    y='Urea_per_ha',
                    title='Distribusi Urea per Hektar (kg/ha)',
                    color='Komoditas',
                    points='outliers'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    df_filtered,
                    x='Komoditas',
                    y='NPK_per_ha',
                    title='Distribusi NPK per Hektar (kg/ha)',
                    color='Komoditas',
                    points='outliers'
                )
                st.plotly_chart(fig, use_container_width=True)

        if 'Organik_per_ha' in df_filtered.columns:
            fig = px.box(
                df_filtered,
                x='Komoditas',
                y='Organik_per_ha',
                title='Distribusi Organik per Hektar (kg/ha)',
                color='Komoditas',
                points='outliers'
            )
            st.plotly_chart(fig, use_container_width=True)


# ==========================================
# PAGE 3: PREDIKSI DATA BARU
# ==========================================
elif page == "Prediksi Data Baru":
    st.markdown('<div class="section-header">Prediksi Data Baru</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Cara Menggunakan:</strong><br>
    1. Masukkan data petani baru di form di bawah<br>
    2. Sistem akan otomatis menghitung status (jika standar aktif)<br>
    3. Rekomendasi akan diberikan berdasarkan analisis ML dan standar (jika aktif)
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        st.subheader("Data Petani Baru")
        
        col1, col2 = st.columns(2)
        
        with col1:
            id_petani = st.text_input("ID Petani", value="P_NEW_001", help="ID unik petani")
            desa = st.text_input("Desa", value="Desa Baru")
            kelompok = st.text_input("Kelompok Tani", value="Kelompok Baru")
        
        with col2:
            # Gunakan komoditas dari config.yaml secara konsisten
            all_commodities_config = get_all_commodities_from_config()
            commodity_options = all_commodities_config if all_commodities_config else ['PADI', 'JAGUNG', 'KEDELAI', 'KOPI', 'CABAI'] # Fallback
            
            komoditas = st.selectbox("Komoditas", commodity_options)
            luas_m2 = st.number_input("Luas Tanah (m²)", min_value=100, max_value=100000, value=5000, step=100)
        
        st.markdown("---")
        st.subheader("Data Pupuk per Musim Tanam (kg)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🌱 Musim Tanam 1**")
            urea_mt1 = st.number_input("Urea MT1 (kg)", min_value=0.0, value=60.0, step=5.0, key="urea1")
            npk_mt1 = st.number_input("NPK MT1 (kg)", min_value=0.0, value=40.0, step=5.0, key="npk1")
            organik_mt1 = st.number_input("Organik MT1 (kg)", min_value=0.0, value=30.0, step=5.0, key="org1")
        
        with col2:
            st.markdown("**🌱 Musim Tanam 2**")
            urea_mt2 = st.number_input("Urea MT2 (kg)", min_value=0.0, value=60.0, step=5.0, key="urea2")
            npk_mt2 = st.number_input("NPK MT2 (kg)", min_value=0.0, value=40.0, step=5.0, key="npk2")
            organik_mt2 = st.number_input("Organik MT2 (kg)", min_value=0.0, value=30.0, step=5.0, key="org2")
        
        with col3:
            st.markdown("**🌱 Musim Tanam 3**")
            urea_mt3 = st.number_input("Urea MT3 (kg)", min_value=0.0, value=50.0, step=5.0, key="urea3")
            npk_mt3 = st.number_input("NPK MT3 (kg)", min_value=0.0, value=30.0, step=5.0, key="npk3")
            organik_mt3 = st.number_input("Organik MT3 (kg)", min_value=0.0, value=25.0, step=5.0, key="org3")
        
        submitted = st.form_submit_button("Analisis Data", use_container_width=True, type="primary")
    
    if submitted:
        # Prepare input data
        input_df = pd.DataFrame([{
            'ID_Petani': id_petani,
            'Desa': desa,
            'Kelompok_Tani': kelompok,
            'Komoditas': komoditas,
            'Luas_Tanah_m2': luas_m2,
            'Urea_MT1': urea_mt1, 'NPK_MT1': npk_mt1, 'Organik_MT1': organik_mt1,
            'Urea_MT2': urea_mt2, 'NPK_MT2': npk_mt2, 'Organik_MT2': organik_mt2,
            'Urea_MT3': urea_mt3, 'NPK_MT3': npk_mt3, 'Organik_MT3': organik_mt3
        }])
        
        try:
            # Preprocess
            processed_df = preprocess_pipeline(
                input_df, 
                standards_manager=standards_manager if standards_enabled else None
            )
            
            if len(processed_df) == 0:
                st.error("❌ Error processing input data")
            else:
                result = processed_df.iloc[0]
                
                # Display results
                st.markdown("---")
                st.markdown("### 📋 Hasil Analisis")
                
                # Basic info
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Luas Lahan", f"{result.get('Luas_Tanah_ha', 0):.2f} ha")
                
                with col2:
                    st.metric("Total Pupuk", f"{result.get('Total_Pupuk', 0):.1f} kg")
                
                with col3:
                    st.metric("Intensitas", f"{result.get('Total_per_ha', 0):.1f} kg/ha")
                
                # Status per jenis pupuk (if standards enabled)
                if standards_enabled:
                    st.markdown("### 🌱 Status per Jenis Pupuk")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        status_urea = result.get('Status_Urea', 'Unknown')
                        if status_urea == 'Overuse':
                            st.markdown('<div class="danger-box"><strong>🌾 Urea:</strong> Overuse 🚨<br>Kurangi dosis pupuk</div>', unsafe_allow_html=True)
                        elif status_urea == 'Underuse':
                            st.markdown('<div class="warning-box"><strong>🌾 Urea:</strong> Underuse ⚠️<br>Tambahkan pupuk</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="success-box"><strong>🌾 Urea:</strong> Normal ✅<br>Sudah sesuai standar</div>', unsafe_allow_html=True)
                    
                    with col2:
                        status_npk = result.get('Status_NPK', 'Unknown')
                        if status_npk == 'Overuse':
                            st.markdown('<div class="danger-box"><strong>🌱 NPK:</strong> Overuse 🚨<br>Kurangi dosis pupuk</div>', unsafe_allow_html=True)
                        elif status_npk == 'Underuse':
                            st.markdown('<div class="warning-box"><strong>🌱 NPK:</strong> Underuse ⚠️<br>Tambahkan pupuk</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="success-box"><strong>🌱 NPK:</strong> Normal ✅<br>Sudah sesuai standar</div>', unsafe_allow_html=True)
                    
                    with col3:
                        status_organik = result.get('Status_Organik', 'Unknown')
                        if status_organik == 'Overuse':
                            st.markdown('<div class="danger-box"><strong>🍂 Organik:</strong> Overuse 🚨<br>Kurangi dosis pupuk</div>', unsafe_allow_html=True)
                        elif status_organik == 'Underuse':
                            st.markdown('<div class="warning-box"><strong>🍂 Organik:</strong> Underuse ⚠️<br>Tambahkan pupuk</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="success-box"><strong>🍂 Organik:</strong> Normal ✅<br>Sudah sesuai standar</div>', unsafe_allow_html=True)
                    
                    # Final status
                    st.markdown("### Status Keseluruhan")
                    final_status = result.get('Final_Status', 'Unknown')
                    
                    if final_status == 'Overuse':
                        st.error(f"🚨 **OVERUSE** - Penggunaan pupuk melebihi standar")
                    elif final_status == 'Underuse':
                        st.warning(f"⚠️ **UNDERUSE** - Penggunaan pupuk di bawah standar")
                    else:
                        st.success(f"✅ **NORMAL** - Penggunaan pupuk sesuai standar")
                else:
                    st.info("ℹ️ Aktifkan standar pupuk untuk melihat status underuse/overuse")
                
                # Detailed info table
                st.markdown("### Detail Penggunaan Pupuk")
                
                detail_data = {
                    'Jenis Pupuk': ['Urea', 'NPK', 'Organik', 'TOTAL'],
                    'Penggunaan (kg)': [
                        result.get('Total_Urea', 0),
                        result.get('Total_NPK', 0),
                        result.get('Total_Organik', 0),
                        result.get('Total_Pupuk', 0)
                    ],
                    'Per Hektar (kg/ha)': [
                        result.get('Urea_per_ha', 0),
                        result.get('NPK_per_ha', 0),
                        result.get('Organik_per_ha', 0),
                        result.get('Total_per_ha', 0)
                    ]
                }
                
                if standards_enabled:
                    detail_data['Standar Min (kg)'] = [
                        result.get('Jatah_Urea_Min', 0),
                        result.get('Jatah_NPK_Min', 0),
                        result.get('Jatah_Organik_Min', 0),
                        '-'
                    ]
                    detail_data['Standar Max (kg)'] = [
                        result.get('Jatah_Urea_Max', 0),
                        result.get('Jatah_NPK_Max', 0),
                        result.get('Jatah_Organik_Max', 0),
                        '-'
                    ]
                    detail_data['Status'] = [
                        result.get('Status_Urea', '-'),
                        result.get('Status_NPK', '-'),
                        result.get('Status_Organik', '-'),
                        result.get('Final_Status', '-')
                    ]
                
                detail_df = pd.DataFrame(detail_data)
                st.dataframe(detail_df, use_container_width=True, hide_index=True)
                
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# ==========================================
# PAGE 4: CLUSTERING & POLA (WITH ENHANCED EXPLANATION)
# ==========================================
elif page == "Clustering & Pola":
    st.markdown('<div class="section-header">Clustering & Pola Penggunaan Pupuk</div>', unsafe_allow_html=True)
    
    if df is None:
        st.error("❌ Data tidak tersedia")
        st.stop()
    
    if 'Cluster_ID' not in df.columns:
        st.warning("⚠️ Data clustering belum tersedia. Jalankan `python main.py` untuk melakukan clustering.")
        st.stop()
    
    st.markdown("""
    <div class="info-box">
    <strong>Apa itu Clustering? (Penjelasan Sederhana)</strong><br><br>
    
    Clustering adalah cara komputer mengelompokkan petani yang memiliki <strong>pola penggunaan pupuk serupa</strong>. 
    Bayangkan seperti mengelompokkan siswa berdasarkan hobi mereka.<br><br>
    
    <strong>Poin Penting:</strong><br>
    <ul style="margin: 10px 0; padding-left: 20px;">
        <li><strong>Cluster BUKAN penilaian baik/buruk</strong> - Ini hanya pengelompokan berdasarkan kesamaan</li>
        <li>Setiap cluster punya <strong>karakteristik unik</strong> - luas lahan, intensitas pupuk, jenis pupuk dominan</li>
        <li><strong>Tidak ada cluster yang "salah"</strong> - Semuanya valid, hanya berbeda pendekatan</li>
        <li>Berguna untuk <strong>strategi distribusi dan edukasi yang lebih tepat sasaran</strong></li>
    </ul>
    
    <em>Contoh: Cluster A = petani lahan kecil dengan pupuk organik tinggi. Cluster B = petani lahan besar dengan urea dominan. 
    Keduanya valid, hanya perlu pendekatan berbeda!</em>
    </div>
    """, unsafe_allow_html=True)
    
    n_clusters = df['Cluster_ID'].nunique()
    
    if n_clusters < 2:
        st.warning(f"""
        ⚠️ <strong>Hanya {n_clusters} cluster ditemukan</strong>
        
        Jumlah cluster terlalu sedikit untuk analisis yang bermakna. 
        Idealnya minimal 3-5 cluster untuk melihat pola yang beragam.
        
        <strong>Penyebab:</strong>
        - Dataset terlalu kecil
        - Pola penggunaan pupuk sangat homogen
        
        <strong>Saran:</strong> Tambahkan lebih banyak data atau sesuaikan parameter clustering.
        """)
    
    st.markdown(f"### Ringkasan: {n_clusters} Kelompok Petani Teridentifikasi")
    
    # Display cluster sizes
    col1, col2, col3, col4 = st.columns(4)
    cluster_counts = df['Cluster_ID'].value_counts().sort_index()
    
    for i, (cluster_id, count) in enumerate(cluster_counts.items()):
        with [col1, col2, col3, col4][i % 4]:
            pct = count / len(df) * 100
            st.metric(
                f"Cluster {cluster_id}",
                f"{count:,} petani",
                f"{pct:.1f}%"
            )
    
    st.markdown("---")
    
    st.markdown("Karakteristik Setiap Cluster (Penjelasan Detail)")
    
    # Get characteristics from models atau generate baru
    if models and 'clustering' in models and 'characteristics' in models['clustering']:
        characteristics = models['clustering']['characteristics']
    else:
        # Import only if needed
        from src.clustering import generate_cluster_characteristics
        characteristics = generate_cluster_characteristics(df)
    
    # Display each cluster dengan ringkasan statistik
    for cluster_id in sorted(df['Cluster_ID'].unique()):
        with st.expander(f"Detail Cluster {cluster_id} ({cluster_counts[cluster_id]:,} petani)", expanded=(cluster_id == 0)):
            cluster_data = df[df['Cluster_ID'] == cluster_id]
            
            if str(cluster_id) in characteristics:
                st.markdown(characteristics[str(cluster_id)], unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Statistik ringkas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_luas = cluster_data['Luas_Tanah_ha'].mean() if 'Luas_Tanah_ha' in cluster_data.columns else 0
                st.metric("Rata-rata Luas", f"{avg_luas:.2f} ha")
            
            with col2:
                avg_urea = cluster_data['Urea_per_ha'].mean() if 'Urea_per_ha' in cluster_data.columns else 0
                st.metric("Rata-rata Urea", f"{avg_urea:.1f} kg/ha")
            
            with col3:
                avg_npk = cluster_data['NPK_per_ha'].mean() if 'NPK_per_ha' in cluster_data.columns else 0
                st.metric("Rata-rata NPK", f"{avg_npk:.1f} kg/ha")
            
            with col4:
                avg_organik = cluster_data['Organik_per_ha'].mean() if 'Organik_per_ha' in cluster_data.columns else 0
                st.metric("Rata-rata Organik", f"{avg_organik:.1f} kg/ha")
            
            if all(col in cluster_data.columns for col in ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha']):
                avg_total = avg_urea + avg_npk + avg_organik
                if avg_total > 0:
                    proportions = {
                        'Urea': (avg_urea / avg_total) * 100,
                        'NPK': (avg_npk / avg_total) * 100,
                        'Organik': (avg_organik / avg_total) * 100
                    }
                    
                    fig = px.pie(
                        values=list(proportions.values()),
                        names=list(proportions.keys()),
                        title=f'Komposisi Pupuk Cluster {cluster_id}',
                        color=list(proportions.keys()),
                        color_discrete_map={'Urea': '#66b5f6', 'NPK': '#81c784', 'Organik': '#ffb74d'}
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### Rekomendasi Strategi per Cluster")
    
    st.markdown("""
    <div class="success-box">
    <strong>Mengapa Penting?</strong><br>
    Setiap cluster memiliki karakteristik berbeda, sehingga memerlukan:
    <ul style="margin: 10px 0; padding-left: 20px;">
        <li><strong>Strategi distribusi yang berbeda</strong> - Cluster lahan besar vs kecil perlu pendekatan berbeda</li>
        <li><strong>Edukasi yang disesuaikan</strong> - Cluster intensitas tinggi perlu edukasi efisiensi, cluster rendah perlu edukasi manfaat</li>
        <li><strong>Monitoring yang tepat sasaran</strong> - Fokuskan sumber daya pada cluster yang paling memerlukan perhatian</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Display recommendations per cluster (simplified)
    for cluster_id in sorted(df['Cluster_ID'].unique()):
        with st.expander(f"Rekomendasi untuk Cluster {cluster_id}", expanded=False):
            cluster_data = df[df['Cluster_ID'] == cluster_id]
            
            # Basic insights based on characteristics
            rec = []
            if 'Total_per_ha' in cluster_data.columns and cluster_data['Total_per_ha'].mean() > 300: # Threshold example
                rec.append("Fokus pada edukasi efisiensi penggunaan pupuk dan pengurangan dosis")
            elif 'Total_per_ha' in cluster_data.columns and cluster_data['Total_per_ha'].mean() < 100: # Threshold example
                rec.append("Tingkatkan edukasi tentang pentingnya pupuk bagi hasil panen yang optimal")
            
            if 'Luas_Tanah_ha' in cluster_data.columns and cluster_data['Luas_Tanah_ha'].mean() > 2: # Threshold example
                rec.append("Pertimbangkan program bantuan yang lebih besar karena luas lahan rata-rata besar")
            else:
                rec.append("Pertimbangkan program bantuan yang lebih terjangkau untuk petani lahan kecil")
            
            if standards_enabled and 'Final_Status' in cluster_data.columns:
                overuse_pct = (cluster_data['Final_Status'] == 'Overuse').mean() * 100
                underuse_pct = (cluster_data['Final_Status'] == 'Underuse').mean() * 100
                
                if overuse_pct > 30:
                    rec.append("Waspadai potensi overuse, lakukan sosialisasi dampak negatifnya")
                elif underuse_pct > 30:
                    rec.append("Dorong penggunaan pupuk sesuai standar untuk hasil maksimal")
            
            if not rec:
                rec.append("Perlu analisis lebih lanjut untuk menentukan strategi spesifik.")
            
            st.markdown("<strong>Saran Strategi:</strong>", unsafe_allow_html=True)
            for item in rec:
                st.markdown(f"- {item}")
    
    st.markdown("---")
    
    # Scatter plot: Luas vs Intensitas colored by cluster
    st.markdown("### Visualisasi: Luas Lahan vs Intensitas Pupuk")
    
    if all(col in df.columns for col in ['Luas_Tanah_ha', 'Total_per_ha', 'Cluster_ID']):
        fig = px.scatter(
            df,
            x='Luas_Tanah_ha',
            y='Total_per_ha',
            color='Cluster_ID',
            title='Pola Luas Lahan vs Intensitas Pupuk per Cluster',
            labels={'Luas_Tanah_ha': 'Luas Lahan (ha)', 'Total_per_ha': 'Intensitas Pupuk (kg/ha)'},
            hover_data=['Komoditas', 'Total_Pupuk'] if 'Komoditas' in df.columns else None
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 5: REKOMENDASI (ENHANCED)
# ==========================================
elif page == "Rekomendasi":
    st.markdown('<div class="section-header">Rekomendasi & Action Plan</div>', unsafe_allow_html=True)
    
    if df is None:
        st.error("❌ Data tidak tersedia")
        st.stop()
    
    if 'Rekomendasi' not in df.columns or 'Prioritas' not in df.columns:
        st.warning("⚠️ Rekomendasi belum tersedia. Jalankan `python main.py` terlebih dahulu.")
        st.stop()
    
    st.markdown("""
    <div class="info-box">
    <strong>Tentang Rekomendasi:</strong><br>
    Rekomendasi dihasilkan berdasarkan:<br>
    1. Deteksi anomali (ML)<br>
    2. Pola clustering<br>
    3. Standar pupuk (jika aktif)<br>
    4. Intensitas penggunaan per hektar
    </div>
    """, unsafe_allow_html=True)
    
    # Summary by priority
    st.subheader("Ringkasan Prioritas")
    
    col1, col2, col3 = st.columns(3)
    
    tinggi = (df['Prioritas'] == 'Tinggi').sum()
    sedang = (df['Prioritas'] == 'Sedang').sum()
    rendah = (df['Prioritas'] == 'Rendah').sum()
    
    with col1:
        st.markdown(f"""
        <div class="danger-box">
        <h3>Prioritas Tinggi</h3>
        <h2>{tinggi:,} petani</h2>
        <p>{tinggi/len(df)*100:.1f}% dari total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="warning-box">
        <h3>Prioritas Sedang</h3>
        <h2>{sedang:,} petani</h2>
        <p>{sedang/len(df)*100:.1f}% dari total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="success-box">
        <h3>Prioritas Rendah</h3>
        <h2>{rendah:,} petani</h2>
        <p>{rendah/len(df)*100:.1f}% dari total</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Detail Rekomendasi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        priority_filter = st.selectbox(
            "Filter berdasarkan prioritas:",
            ['Semua', 'Tinggi', 'Sedang', 'Rendah']
        )
    
    with col2:
        if 'Cluster_ID' in df.columns:
            cluster_options = ['Semua'] + [f"Cluster {i}" for i in sorted(df['Cluster_ID'].unique())]
            cluster_filter = st.selectbox(
                "Filter berdasarkan cluster:",
                cluster_options
            )
        else:
            cluster_filter = 'Semua'
    
    with col3:
        if 'Komoditas' in df.columns:
            komoditas_options = ['Semua'] + sorted(df['Komoditas'].unique().tolist())
            komoditas_filter_rec = st.selectbox(
                "Filter berdasarkan komoditas:",
                komoditas_options,
                key='komoditas_rec'
            )
        else:
            komoditas_filter_rec = 'Semua'
    
    df_filtered = df.copy()
    
    if priority_filter != 'Semua':
        df_filtered = df_filtered[df_filtered['Prioritas'] == priority_filter]
    
    if cluster_filter != 'Semua' and 'Cluster_ID' in df.columns:
        cluster_id = int(cluster_filter.split()[-1])
        df_filtered = df_filtered[df_filtered['Cluster_ID'] == cluster_id]
    
    if komoditas_filter_rec != 'Semua' and 'Komoditas' in df.columns:
        df_filtered = df_filtered[df_filtered['Komoditas'] == komoditas_filter_rec]
    
    st.info(f"Menampilkan {len(df_filtered):,} petani")
    
    # Display recommendations
    display_cols = ['ID_Petani', 'Desa', 'Komoditas', 'Luas_Tanah_ha', 'Total_per_ha', 'Prioritas', 'Rekomendasi', 'Action_Plan']
    if standards_enabled and 'Final_Status' in df_filtered.columns:
        display_cols.insert(6, 'Final_Status')
    
    if 'Anomaly_Label' in df_filtered.columns:
        display_cols.insert(6, 'Anomaly_Label')
    
    display_cols = [col for col in display_cols if col in df_filtered.columns]
    
    st.dataframe(df_filtered[display_cols], use_container_width=True, height=500)
    
    # Download recommendations
    csv = df_filtered[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Rekomendasi (CSV)",
        data=csv,
        file_name=f"rdkk_rekomendasi_{priority_filter.lower()}.csv",
        mime="text/csv",
    )
    
    # Action plan summary
    if 'Action_Plan' in df_filtered.columns:
        st.markdown("---")
        st.subheader("Action Plan Prioritas Tinggi")
        
        high_priority = df_filtered[df_filtered['Prioritas'] == 'Tinggi']
        
        if len(high_priority) > 0:
            st.markdown(f"**{len(high_priority):,} petani memerlukan tindakan segera:**")
            
            # Count action types
            action_counts = {}
            for actions in high_priority['Action_Plan']:
                if pd.notna(actions):
                    for action in str(actions).split(';'):
                        action = action.strip()
                        if action:
                            action_counts[action] = action_counts.get(action, 0) + 1
            
            if action_counts:
                sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
                
                for action, count in sorted_actions[:5]:
                    st.markdown(f"- **{action}**: {count} petani")
        else:
            st.success("✅ Tidak ada petani dengan prioritas tinggi saat ini")

# ==========================================
# PAGE 6: TENTANG MODEL (HIGHLY ENHANCED)
# ==========================================
elif page == "Tentang Model":
    st.markdown('<div class="section-header">Tentang Model Machine Learning</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
    <strong>Penjelasan untuk User Awam:</strong><br>
    Halaman ini menjelaskan bagaimana sistem menggunakan kecerdasan buatan (AI) untuk menganalisis data pupuk subsidi.
    Tidak perlu keahlian teknis - kami jelaskan dengan bahasa sederhana!
    </div>
    """, unsafe_allow_html=True)
    
    # Section 1: Ringkasan Model
    st.markdown("---")
    st.markdown("1. Ringkasan Model AI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <strong>Isolation Forest (Deteksi Anomali)</strong>
        
        Model ini bekerja seperti "pendeteksi pola aneh":
        - Menganalisis semua data penggunaan pupuk
        - Mencari pola yang tidak biasa atau ekstrem
        - Menandai petani dengan penggunaan yang sangat berbeda dari mayoritas
        - Bekerja otomatis tanpa perlu aturan manual
        
        <strong>Kenapa penting?</strong>  
        Membantu menemukan kasus overuse atau underuse yang signifikan secara objektif berdasarkan data aktual.
        """, unsafe_allow_html=True)
        
        if df is not None and 'Anomaly_Label' in df.columns:
            anomaly_counts = df['Anomaly_Label'].value_counts()
            normal_count = anomaly_counts.get('Normal', 0)
            anomaly_count = len(df) - normal_count
            
            st.metric("Total Data Dianalisis", f"{len(df):,}")
            st.metric("Deteksi Normal", f"{normal_count:,}", f"{normal_count/len(df)*100:.1f}%")
            st.metric("Deteksi Anomali", f"{anomaly_count:,}", f"{anomaly_count/len(df)*100:.1f}%")
    
    with col2:
        st.markdown(f"""
        <strong>KMeans Clustering (Pengelompokan)</strong>
        
        Model ini seperti "pengelompokan otomatis":
        - Mengelompokkan petani dengan pola serupa
        - Berdasarkan luas lahan, intensitas pupuk, dan pola penggunaan
        - Membantu memahami segmen petani yang berbeda
        - Berguna untuk strategi distribusi yang lebih tepat sasaran
        
        <strong>Kenapa penting?</strong>  
        Membantu pemerintah memahami berbagai tipe petani dan kebutuhan mereka yang berbeda-beda.
        """, unsafe_allow_html=True)
        
        if df is not None and 'Cluster_ID' in df.columns:
            n_clusters = df['Cluster_ID'].nunique()
            st.metric("Jumlah Kelompok", f"{n_clusters} cluster")
            
            if models and 'clustering' in models:
                feature_count = len(models['clustering'].get('feature_cols', []))
                st.metric("Fitur yang Dianalisis", f"{feature_count} variabel")
    
    # Section 2: Pipeline Diagram
    st.markdown("---")
    st.markdown("2. Alur Kerja Sistem (Pipeline)")
    
    st.markdown("""
    Berikut adalah tahapan pemrosesan data dari awal hingga menghasilkan rekomendasi:
    """)
    
    pipeline_cols = st.columns(7)
    
    pipeline_steps = [
        ("1. Load Data", "Membaca data CSV petani", "#e3f2fd"),
        ("2. Preprocessing", "Membersihkan & standarisasi data", "#f3e5f5"),
        ("3. Feature Engineering", "Menghitung pupuk per ha, total, dll", "#e8f5e9"),
        ("4. Anomaly Detection", "Deteksi pola tidak normal (ML)", "#fff3e0"),
        ("5. Clustering", "Pengelompokan pola serupa", "#fce4ec"),
        ("6. Generate Recommendations", "Buat rekomendasi & prioritas", "#e0f2f1"),
        ("7. Export Results", "Simpan hasil analisis", "#f1f8e9")
    ]
    
    for i, (title, desc, color) in enumerate(pipeline_steps):
        with pipeline_cols[i]:
            st.markdown(f"""
            <div style="
            background: {color};
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            ">
            <strong>{title}</strong>
            <p style="font-size: 0.8rem; margin-top: 0.5rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    
    # Section 3: Cara Model Mengambil Keputusan
    st.markdown("---")
    st.markdown("3. Cara Model Mengambil Keputusan")
    
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["Anomaly Detection", "Clustering", "Rekomendasi"])
        
        with tab1:
            st.markdown(f"""
            <strong>Bagaimana Isolation Forest Mendeteksi Anomali?</strong>
            
            Model ini menggunakan konsep "isolasi":
            1. Data normal biasanya berkelompok dan sulit dipisahkan
            2. Data anomali biasanya terisolasi dan mudah dipisahkan
            3. Model menghitung "skor anomali" untuk setiap petani
            4. Skor tinggi = kemungkinan besar anomali (pola tidak biasa)
            """,unsafe_allow_html=True)
            
            if 'Anomaly_Score' in df.columns:
                st.markdown("##### Distribusi Skor Anomali")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.histogram(
                        df,
                        x='Anomaly_Score',
                        nbins=50,
                        title='Distribusi Skor Anomali (lebih tinggi = lebih anomali)',
                        color_discrete_sequence=['#ff7043'],
                        labels={'Anomaly_Score': 'Skor Anomali', 'count': 'Jumlah Petani'}
                    )
                    fig.add_vline(x=-0.1, line_dash="dash", line_color="red", 
                                 annotation_text="Threshold Anomali", annotation_position="top")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("Rata-rata Skor", f"{df['Anomaly_Score'].mean():.3f}")
                    st.metric("Skor Tertinggi", f"{df['Anomaly_Score'].max():.3f}", help="Paling anomali")
                    st.metric("Skor Terendah", f"{df['Anomaly_Score'].min():.3f}", help="Paling normal")
            
            if 'Anomaly_Label' in df.columns:
                st.markdown("##### Kategori Anomali")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    anomaly_dist = df['Anomaly_Label'].value_counts()
                    fig = px.pie(
                        values=anomaly_dist.values,
                        names=anomaly_dist.index,
                        title='Distribusi Kategori Anomali',
                        color_discrete_sequence=['#4caf50', '#ff9800', '#f44336']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown(f"<strong>Penjelasan Kategori:</strong>", unsafe_allow_html=True)
                    for label in anomaly_dist.index:
                        count = anomaly_dist[label]
                        pct = count / len(df) * 100
                        
                    if label == 'Normal':
                        st.markdown(
                            f"<strong>{label}</strong>: {count:,} petani ({pct:.1f}%)",
                            unsafe_allow_html=True
                        )
                        st.markdown("Penggunaan pupuk dalam batas wajar")

                    elif label == 'Ringan':
                        st.markdown(
                            f"<strong>{label}</strong>: {count:,} petani ({pct:.1f}%)",
                            unsafe_allow_html=True
                        )
                        st.markdown("Sedikit menyimpang, perlu perhatian")

                    else:
                        st.markdown(
                            f"<strong>{label}</strong>: {count:,} petani ({pct:.1f}%)",
                            unsafe_allow_html=True
                        )
                        st.markdown("Sangat menyimpang, prioritas tinggi")

        
        with tab2:
            st.markdown("""
            <strong>Bagaimana KMeans Mengelompokkan Petani?</strong>
            
            Model ini mengelompokkan berdasarkan kesamaan:
            1. Menghitung "jarak" antar petani berdasarkan fitur-fitur tertentu
            2. Petani dengan karakteristik mirip dikelompokkan bersama
            3. Setiap cluster memiliki "pusat" yang merepresentasikan rata-rata kelompok
            4. Membantu memahami segmen-segmen petani yang berbeda
            """)
            
            if 'Cluster_ID' in df.columns and models and 'clustering' in models:
                st.markdown("##### Visualisasi Cluster dalam 2D (PCA)")
                
                feature_cols = models['clustering'].get('feature_cols', [])
                available_features = [f for f in feature_cols if f in df.columns]
                
                if len(available_features) >= 2:
                    try:
                        from sklearn.decomposition import PCA
                        
                        # Prepare data for PCA
                        X = df[available_features].fillna(0)
                        
                        # Apply PCA
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X)
                        
                        # Create visualization dataframe
                        viz_df = pd.DataFrame({
                            'PC1': X_pca[:, 0],
                            'PC2': X_pca[:, 1],
                            'Cluster': df['Cluster_ID'].astype(str),
                            'Total_per_ha': df['Total_per_ha'] if 'Total_per_ha' in df.columns else 0,
                            'Komoditas': df['Komoditas'] if 'Komoditas' in df.columns else 'N/A'
                        })
                        
                        fig = px.scatter(
                            viz_df,
                            x='PC1',
                            y='PC2',
                            color='Cluster',
                            title='Visualisasi Cluster dengan PCA (Principal Component Analysis)',
                            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
                            hover_data=['Komoditas', 'Total_per_ha'],
                            size='Total_per_ha' if 'Total_per_ha' in df.columns else None,
                            size_max=10
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info(f"""
                        <strong>Penjelasan Visualisasi:</strong>
                        - PCA mereduksi {len(available_features)} dimensi menjadi 2D untuk visualisasi
                        - Titik yang berdekatan = petani dengan pola serupa
                        - Warna berbeda = cluster berbeda
                        - Variance explained: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%
                        """)
                    except Exception as e:
                        st.warning(f"Tidak dapat membuat visualisasi PCA: {e}")
                else:
                    # Fallback to simple 2D scatter
                    if all(col in df.columns for col in ['Luas_Tanah_ha', 'Total_per_ha']):
                        fig = px.scatter(
                            df,
                            x='Luas_Tanah_ha',
                            y='Total_per_ha',
                            color='Cluster_ID',
                            title='Visualisasi Cluster: Luas Lahan vs Intensitas Pupuk',
                            labels={
                                'Luas_Tanah_ha': 'Luas Lahan (ha)',
                                'Total_per_ha': 'Intensitas Pupuk (kg/ha)',
                                'Cluster_ID': 'Cluster'
                            },
                            hover_data=['Komoditas', 'Total_Pupuk'] if 'Komoditas' in df.columns else None,
                            size='Total_Pupuk' if 'Total_Pupuk' in df.columns else None,
                            size_max=15
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("##### Karakteristik Setiap Cluster")
                
                for cluster_id in sorted(df['Cluster_ID'].unique()):
                    cluster_data = df[df['Cluster_ID'] == cluster_id]
                    
                    with st.expander(f"Cluster {cluster_id} ({cluster_data['Cluster_ID'].count():,} petani)", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Luas Rata-rata", f"{cluster_data['Luas_Tanah_ha'].mean():.2f} ha")
                        
                        with col2:
                            if 'Total_per_ha' in cluster_data.columns:
                                st.metric("Intensitas Rata-rata", f"{cluster_data['Total_per_ha'].mean():.1f} kg/ha")
                        
                        with col3:
                            if 'Total_Pupuk' in cluster_data.columns:
                                st.metric("Total Pupuk Rata-rata", f"{cluster_data['Total_Pupuk'].mean():.1f} kg")
                        
                        # Pupuk breakdown
                        if all(col in cluster_data.columns for col in ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha']):
                            pupuk_avg = {
                                'Urea': cluster_data['Urea_per_ha'].mean(),
                                'NPK': cluster_data['NPK_per_ha'].mean(),
                                'Organik': cluster_data['Organik_per_ha'].mean()
                            }
                            
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=list(pupuk_avg.keys()),
                                    y=list(pupuk_avg.values()),
                                    marker_color=['#64b5f6', '#81c784', '#ffb74d']
                                )
                            ])
                            fig.update_layout(
                                title=f"Rata-rata Pupuk per Hektar - Cluster {cluster_id}",
                                yaxis_title="kg/ha",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("""
            <strong>Bagaimana Sistem Menghasilkan Rekomendasi?</strong>
            
            Rekomendasi dihasilkan dari kombinasi:
            1. <strong>Hasil Deteksi Anomali ML</strong> → Prioritas tertinggi untuk anomali berat
            2. <strong>Perbandingan dengan Median</strong> → Identifikasi overuse/underuse signifikan
            3. <strong>Karakteristik Cluster</strong> → Rekomendasi disesuaikan dengan tipe petani
            4. <strong>Standar Pupuk (jika aktif)</strong> → Validasi tambahan berdasarkan aturan
            
            <strong>Tingkat Prioritas:</strong>
            - 🚨 <strong>Tinggi</strong>: Anomali berat atau overuse sangat signifikan (>50% dari median)
            - ⚠️ <strong>Sedang</strong>: Anomali ringan atau overuse/underuse moderat (30-50%)
            - ✅ <strong>Rendah</strong>: Pola normal dengan sedikit penyimpangan (<30%)
            """)
            
            if 'Prioritas' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    priority_dist = df['Prioritas'].value_counts()
                    fig = px.pie(
                        values=priority_dist.values,
                        names=priority_dist.index,
                        title='Distribusi Prioritas Rekomendasi',
                        color=priority_dist.index,
                        color_discrete_map={'Tinggi': '#f44336', 'Sedang': '#ff9800', 'Rendah': '#4caf50'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("##### Statistik Prioritas")
                    
                    for priority in ['Tinggi', 'Sedang', 'Rendah']:
                        if priority in priority_dist.index:
                            count = priority_dist[priority]
                            pct = count / len(df) * 100
                            
                            if priority == 'Tinggi':
                                st.markdown(f"""
                                <div class="danger-box">
                                <strong>🚨 {priority}:</strong> {count:,} petani ({pct:.1f}%)<br>
                                Perlu tindakan segera
                                </div>
                                """, unsafe_allow_html=True)
                            elif priority == 'Sedang':
                                st.markdown(f"""
                                <div class="warning-box">
                                <strong>⚠️ {priority}:</strong> {count:,} petani ({pct:.1f}%)<br>
                                Perlu monitoring
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="success-box">
                                <strong>✅ {priority}:</strong> {count:,} petani ({pct:.1f}%)<br>
                                Kondisi baik
                                </div>
                                """, unsafe_allow_html=True)
        
    # Section 4: Fitur Penting
    st.markdown("---")
    st.markdown("### 4. Fitur-Fitur Penting yang Dianalisis")
    
    st.markdown("""
    Model menganalisis berbagai variabel untuk menghasilkan insight. Berikut adalah fitur-fitur utama:
    """)
    
    if models and 'clustering' in models:
        feature_cols = models['clustering'].get('feature_cols', [])
        
        if feature_cols:
            st.markdown("##### Daftar Fitur yang Digunakan:")
            
            # Categorize features
            feature_categories = {
                'Intensitas Pupuk per Hektar': [],
                'Total Penggunaan': [],
                'Proporsi & Rasio': [],
                'Karakteristik Lahan': [],
                'Lainnya': []
            }
            
            for feat in feature_cols:
                if '_per_ha' in feat:
                    feature_categories['Intensitas Pupuk per Hektar'].append(feat)
                elif 'Total_' in feat or feat in ['Total_Pupuk']:
                    feature_categories['Total Penggunaan'].append(feat)
                elif 'Prop_' in feat or 'Ratio_' in feat or 'Persen_' in feat:
                    feature_categories['Proporsi & Rasio'].append(feat)
                elif 'Luas' in feat:
                    feature_categories['Karakteristik Lahan'].append(feat)
                else:
                    feature_categories['Lainnya'].append(feat)
            
            col1, col2 = st.columns(2)
            
            with col1:
                for category, features in list(feature_categories.items())[:3]:
                    if features:
                        st.markdown(f"<strong>{category}:</strong>", unsafe_allow_html=True)
                        for feat in features:
                            st.markdown(f"- `{feat}`")
                        st.markdown("")
            
            with col2:
                for category, features in list(feature_categories.items())[3:]:
                    if features:
                        st.markdown(f"<strong>{category}:</strong>", unsafe_allow_html=True)
                        for feat in features:
                            st.markdown(f"- `{feat}`")
                        st.markdown("")
    
    if df is not None:
        st.markdown("##### Statistik Deskriptif Fitur Utama")
        
        key_features = ['Luas_Tanah_ha', 'Total_Pupuk', 'Total_per_ha', 'Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha']
        available_features = [f for f in key_features if f in df.columns]
        
        if available_features:
            stats_df = df[available_features].describe().T
            stats_df['median'] = df[available_features].median()
            stats_df = stats_df[['mean', 'median', 'std', 'min', 'max']]
            stats_df.columns = ['Rata-rata', 'Median', 'Std Dev', 'Min', 'Max']
            
            st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
    
    # Section 5: Statistik Model (ENHANCED)
    st.markdown("---")
    st.markdown("5. Statistik & Performa Model")
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("##### Data")
            st.metric("Total Sampel", f"{len(df):,}")
            
            if 'Komoditas' in df.columns:
                st.metric("Jenis Komoditas", df['Komoditas'].nunique())
            
            if 'Desa' in df.columns:
                st.metric("Jumlah Desa", df['Desa'].nunique())
        
        with col2:
            st.markdown("##### Clustering")
            
            if 'Cluster_ID' in df.columns:
                st.metric("Jumlah Cluster", df['Cluster_ID'].nunique())
                
                # Calculate distribution balance
                cluster_counts = df['Cluster_ID'].value_counts()
                balance_score = (cluster_counts.min() / cluster_counts.max()) * 100 if cluster_counts.max() > 0 else 0
                st.metric("Keseimbangan Cluster", f"{balance_score:.1f}%", 
                         help="Semakin tinggi = distribusi semakin merata")
                
                # Silhouette score estimation (simplified)
                if 'Total_per_ha' in df.columns and 'Luas_Tanah_ha' in df.columns:
                    try:
                        from sklearn.metrics import silhouette_score
                        X_simple = df[['Luas_Tanah_ha', 'Total_per_ha']].fillna(0)
                        sil_score = silhouette_score(X_simple, df['Cluster_ID'])
                        st.metric("Silhouette Score", f"{sil_score:.3f}",
                                 help="Range: -1 to 1. >0.5 = good clustering")
                    except:
                        pass
        
        with col3:
            st.markdown("##### Anomaly")
            
            if 'Anomaly_Label' in df.columns:
                anomaly_rate = (df['Anomaly_Label'] != 'Normal').mean() * 100
                st.metric("Tingkat Anomali", f"{anomaly_rate:.1f}%")
                
                if 'Anomaly_Score' in df.columns:
                    avg_score = df['Anomaly_Score'].mean()
                    st.metric("Rata-rata Skor", f"{avg_score:.3f}",
                             help="Skor anomali rata-rata")
                    
                    # Anomali berat count
                    berat_count = (df['Anomaly_Label'] == 'Berat').sum()
                    st.metric("Anomali Berat", f"{berat_count:,}",
                             help="Kasus yang perlu perhatian serius")
        
        with col4:
            st.markdown("##### Deviasi Pupuk")
            
            if 'Total_per_ha' in df.columns:
                median_intensity = df['Total_per_ha'].median()
                mean_intensity = df['Total_per_ha'].mean()
                std_intensity = df['Total_per_ha'].std()
                
                st.metric("Median Intensitas", f"{median_intensity:.1f} kg/ha")
                st.metric("Rata-rata Intensitas", f"{mean_intensity:.1f} kg/ha")
                st.metric("Std Deviasi", f"{std_intensity:.1f} kg/ha",
                         help="Variasi dari rata-rata")
    
    # Section 6: Kesimpulan Otomatis (ENHANCED)
    st.markdown("---")
    st.markdown("### 6. Kesimpulan & Insight Otomatis")
    
    if df is not None:
        st.markdown("<strong>Berdasarkan analisis data, sistem menghasilkan insight berikut:</strong>", unsafe_allow_html=True)
        
        insights = []
        
        # Insight 1: Anomaly rate
        if 'Anomaly_Label' in df.columns:
            anomaly_count = (df['Anomaly_Label'] != 'Normal').sum()
            anomaly_pct = anomaly_count / len(df) * 100
            
            if anomaly_pct > 20:
                insights.append(f"<strong>Tingkat anomali cukup tinggi ({anomaly_pct:.1f}%)</strong> - Perlu investigasi mendalam pada {anomaly_count:,} petani yang terdeteksi memiliki pola penggunaan tidak normal.")
            elif anomaly_pct > 10:
                insights.append(f"<strong>Tingkat anomali moderat ({anomaly_pct:.1f}%)</strong> - Ada {anomaly_count:,} petani yang perlu perhatian khusus.")
            else:
                insights.append(f"<strong>Tingkat anomali rendah ({anomaly_pct:.1f}%)</strong> - Sebagian besar petani ({len(df)-anomaly_count:,}) memiliki pola penggunaan normal.")
        
        # Insight 2: Clustering distribution
        if 'Cluster_ID' in df.columns:
            n_clusters = df['Cluster_ID'].nunique()
            insights.append(f"<strong>Teridentifikasi {n_clusters} kelompok petani berbeda</strong> - Menunjukkan keberagaman pola penggunaan pupuk yang perlu pendekatan berbeda.")
            
            # Most common cluster
            most_common_cluster = df['Cluster_ID'].mode()[0]
            most_common_count = (df['Cluster_ID'] == most_common_cluster).sum()
            insights.append(f"<strong>Cluster {most_common_cluster} adalah yang terbesar</strong> dengan {most_common_count:,} petani ({most_common_count/len(df)*100:.1f}%), menunjukkan pola dominan.")
        
        # Insight 3: Priority distribution
        if 'Prioritas' in df.columns:
            high_priority = (df['Prioritas'] == 'Tinggi').sum()
            high_pct = high_priority / len(df) * 100
            
            if high_pct > 15:
                insights.append(f"<strong>{high_priority:,} petani ({high_pct:.1f}%) memerlukan tindakan segera</strong> - Prioritas tinggi untuk intervensi.")
            elif high_pct > 5:
                insights.append(f"<strong>{high_priority:,} petani ({high_pct:.1f}%) perlu perhatian</strong> - Monitoring lebih intensif diperlukan.")
            else:
                insights.append(f"<strong>Mayoritas petani dalam kondisi baik</strong> - Hanya {high_priority:,} ({high_pct:.1f}%) yang perlu tindakan segera.")
        
        # Insight 4: Commodity-specific
        if 'Komoditas' in df.columns and 'Total_per_ha' in df.columns:
            komoditas_intensity = df.groupby('Komoditas')['Total_per_ha'].mean().sort_values(ascending=False)
            highest_komoditas = komoditas_intensity.index[0]
            highest_value = komoditas_intensity.iloc[0]
            lowest_komoditas = komoditas_intensity.index[-1]
            lowest_value = komoditas_intensity.iloc[-1]
            
            insights.append(f"<strong>{highest_komoditas} memiliki intensitas pupuk tertinggi</strong> ({highest_value:.1f} kg/ha), sedangkan <strong>{lowest_komoditas} terendah</strong> ({lowest_value:.1f} kg/ha).")
        
        # Insight 5: Standards compliance (if enabled)
        if standards_enabled and 'Final_Status' in df.columns:
            overuse_count = (df['Final_Status'] == 'Overuse').sum()
            underuse_count = (df['Final_Status'] == 'Underuse').sum()
            normal_count = (df['Final_Status'] == 'Normal').sum()
            overuse_pct = overuse_count / len(df) * 100
            underuse_pct = underuse_count / len(df) * 100
            normal_pct = normal_count / len(df) * 100
            
            if overuse_pct > underuse_pct:
                insights.append(f"<strong>Overuse lebih dominan ({overuse_pct:.1f}%) dibanding underuse ({underuse_pct:.1f}%)</strong> - Fokus pada edukasi pengurangan dosis dan efisiensi penggunaan.")
            elif underuse_pct > overuse_pct:
                insights.append(f"<strong>Underuse lebih dominan ({underuse_pct:.1f}%) dibanding overuse ({overuse_pct:.1f}%)</strong> - Fokus pada peningkatan akses dan edukasi manfaat pupuk optimal.")
            else:
                insights.append(f"<strong>Distribusi relatif seimbang</strong>: Normal {normal_pct:.1f}%, Overuse {overuse_pct:.1f}%, Underuse {underuse_pct:.1f}%")
        
        # Insight 6: Data quality
        if 'Total_Pupuk' in df.columns:
            zero_pupuk = (df['Total_Pupuk'] == 0).sum()
            if zero_pupuk > 0:
                insights.append(f"<strong>{zero_pupuk:,} petani memiliki total pupuk = 0</strong> - Perlu verifikasi data atau memang tidak menerima pupuk.")
        
        # Display insights
        for i, insight in enumerate(insights, 1):
            st.markdown(f"{i}. {insight}", unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### Rekomendasi Umum Berdasarkan Analisis")
        
        recommendations_general = []
        
        if 'Anomaly_Label' in df.columns:
            anomaly_pct = (df['Anomaly_Label'] != 'Normal').mean() * 100
            if anomaly_pct > 15:
                recommendations_general.append("1. <strong>Lakukan audit mendalam</strong> pada petani dengan anomali berat untuk identifikasi penyebab")
                recommendations_general.append("2. <strong>Tingkatkan monitoring</strong> distribusi pupuk subsidi di lapangan")
        
        if 'Final_Status' in df.columns:
            overuse_pct = (df['Final_Status'] == 'Overuse').mean() * 100
            if overuse_pct > 20:
                recommendations_general.append("3. <strong>Edukasi efisiensi pupuk</strong> - Banyak petani menggunakan pupuk berlebihan")
                recommendations_general.append("4. <strong>Sosialisasi dampak negatif overuse</strong> terhadap lingkungan dan biaya")
        
        if 'Cluster_ID' in df.columns:
            recommendations_general.append("5. <strong>Strategi distribusi berbeda per cluster</strong> - {n_clusters} segmen petani memerlukan pendekatan berbeda".format(n_clusters=df['Cluster_ID'].nunique()))
        
        recommendations_general.append("6. <strong>Update data berkala</strong> untuk meningkatkan akurasi prediksi model")
        
        for rec in recommendations_general:
            st.markdown(rec, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <strong>💬 Catatan Penting:</strong><br>
        Model machine learning terus belajar dari data. Semakin banyak data berkualitas yang diberikan,
        semakin akurat prediksi dan rekomendasi yang dihasilkan. Pastikan data input selalu update dan valid!
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 7: KELOLA STANDAR (WITH CONSISTENT COMMODITIES)
# ==========================================
elif page == "Kelola Standar":
    st.markdown('<div class="section-header">Kelola Standar Pupuk</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Tentang Standar:</strong><br>
    Standar pupuk digunakan untuk menentukan apakah petani mengalami <strong>Overuse</strong>, <strong>Underuse</strong>, atau <strong>Normal</strong>.
    <br><br>
    • Standar dapat diubah sewaktu-waktu sesuai kebijakan<br>
    • Perubahan standar akan <strong>langsung berlaku</strong> di semua halaman (Dashboard, Data Explorer, dll)<br>
    • Format: Min-Max (kg/ha) untuk setiap jenis pupuk<br>
    • Komoditas valid: <strong>{}</strong>
    </div>
    """.format(', '.join(sorted(get_all_commodities_from_config()))), unsafe_allow_html=True)
    
    all_standards = standards_manager.get_all_standards()
    st.markdown(f"### Standar Terdaftar: **{len(all_standards)}** komoditas")
    
    if all_standards:
        # Convert to dataframe for display
        standards_list = []
        for komoditas, std in all_standards.items():
            standards_list.append({
                'Komoditas': komoditas,
                'Urea Min (kg/ha)': std['Urea']['min'],
                'Urea Max (kg/ha)': std['Urea']['max'],
                'NPK Min (kg/ha)': std['NPK']['min'],
                'NPK Max (kg/ha)': std['NPK']['max'],
                'Organik Min (kg/ha)': std['Organik']['min'],
                'Organik Max (kg/ha)': std['Organik']['max']
            })
        
        standards_df = pd.DataFrame(standards_list)
        st.dataframe(standards_df, use_container_width=True, hide_index=True)
        
        if df is not None and 'Komoditas' in df.columns:
            data_commodities = set(df['Komoditas'].dropna().unique())
            standard_commodities = set(all_standards.keys())
            missing_standards = data_commodities - standard_commodities
            
            if missing_standards:
                st.warning(f"""
                **Perhatian:** Ada {len(missing_standards)} komoditas di data yang belum memiliki standar:
                {', '.join(sorted(missing_standards))}
                
                Petani dengan komoditas ini tidak akan mendapat status Underuse/Overuse.
                """)
    else:
        st.warning("⚠️ Belum ada standar pupuk yang terdaftar")
    
    # Form untuk tambah/edit standar
    st.markdown("---")
    st.subheader("Tambah/Edit Standar")
    
    with st.form("standard_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            komoditas = st.text_input("Komoditas", placeholder="Contoh: PADI").upper()
        
        with col2:
            st.write("")  # Spacing
        
        st.markdown("<strong>Standar Pupuk (kg/ha):</strong>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("🌾 <strong>Urea</strong>", unsafe_allow_html=True)
            urea_min = st.number_input("Min", min_value=0.0, value=200.0, step=10.0, key="urea_min")
            urea_max = st.number_input("Max", min_value=0.0, value=300.0, step=10.0, key="urea_max")
        
        with col2:
            st.markdown("🌱 <strong>NPK</strong>", unsafe_allow_html=True)
            npk_min = st.number_input("Min", min_value=0.0, value=150.0, step=10.0, key="npk_min")
            npk_max = st.number_input("Max", min_value=0.0, value=250.0, step=10.0, key="npk_max")
        
        with col3:
            st.markdown("🍂 <strong>Organik</strong>", unsafe_allow_html=True)
            organik_min = st.number_input("Min", min_value=0.0, value=200.0, step=10.0, key="org_min")
            organik_max = st.number_input("Max", min_value=0.0, value=500.0, step=10.0, key="org_max")
        
        submitted = st.form_submit_button("💾 Simpan Standar", type="primary", use_container_width=True)
    
    if submitted:
        if not komoditas:
            st.error("❌ Nama komoditas harus diisi")
        elif urea_min >= urea_max or npk_min >= npk_max or organik_min >= organik_max:
            st.error("❌ Nilai Min harus lebih kecil dari Max")
        else:
            try:
                standards_manager.set_standard(
                    komoditas, 
                    urea_min, urea_max,
                    npk_min, npk_max,
                    organik_min, organik_max
                )
                
                if standards_manager.save_standards():
                    st.success(f"✅ Standar untuk {komoditas} berhasil disimpan!")
                    st.rerun()
                else:
                    st.error("❌ Gagal menyimpan standar ke config.yaml")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Hapus standar
    if all_standards:
        st.markdown("---")
        st.subheader("Hapus Standar")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            komoditas_to_delete = st.selectbox(
                "Pilih komoditas untuk dihapus:",
                list(all_standards.keys())
            )
        
        with col2:
            st.write("")  # Spacing
            if st.button("Hapus", type="secondary", use_container_width=True):
                if standards_manager.delete_standard(komoditas_to_delete):
                    if standards_manager.save_standards():
                        st.success(f"✅ Standar {komoditas_to_delete} berhasil dihapus!")
                        st.rerun()
                    else:
                        st.error("❌ Gagal menyimpan perubahan")
                else:
                    st.error("❌ Gagal menghapus standar")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>RDKK System v4.1</strong> | Powered by Machine Learning & Data Science</p>
    <p>© 2025 - Sistem Analisis Pupuk Subsidi</p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">
        🌾 Membantu petani mengoptimalkan penggunaan pupuk untuk hasil panen yang lebih baik
    </p>
</div>
""", unsafe_allow_html=True)