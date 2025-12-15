"""
RDKK Streamlit Dashboard - Interactive Dashboard
Complete dashboard dengan toggle standar pupuk, prediksi data baru, dan visualisasi lengkap
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

from src.anomaly_explain import get_anomaly_explanation, get_anomaly_comparison, calculate_median_and_std
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

# Initialize session state
if 'standards_manager' not in st.session_state:
    st.session_state.standards_manager = get_standard_manager()
if 'standards_enabled' not in st.session_state:
    st.session_state.standards_enabled = config.get('standar_pupuk', {}).get('enabled', True)

standards_manager = st.session_state.standards_manager

st.markdown("""
<style>
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
        margin-bottom: 2rem;
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

st.markdown('<div class="main-header">🌾 RDKK - Sistem Analisis Pupuk Subsidi</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dashboard Interaktif untuk Deteksi Over/Under Use Pupuk dengan Machine Learning</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    standards_enabled = st.toggle(
        "🎚️ Aktifkan Standar Pupuk", 
        value=st.session_state.standards_enabled,
        help="Toggle ON: Analisis menggunakan standar pupuk per komoditas\nToggle OFF: Analisis berbasis data aktual saja"
    )
    if standards_enabled != st.session_state.standards_enabled:
        st.session_state.standards_enabled = standards_enabled
        st.rerun()

if standards_enabled:
    st.success("✅ Mode: Analisis dengan standar pupuk aktif")
else:
    st.info("ℹ️ Mode: Analisis berbasis data aktual (tanpa standar)")

st.markdown("---")

with st.sidebar:
    # st.image("https://via.placeholder.com/150x50/2e7d32/FFFFFF?text=RDKK", use_container_width=True)
    st.title("📊 Menu Navigasi")
    st.markdown("---")
    
    page = st.radio(
        "Pilih Halaman:",
        [
            "🏠 Dashboard Utama",
            "📊 Data Explorer", 
            "🔍 Prediksi Data Baru",
            "🎯 Clustering & Pola",
            "💡 Rekomendasi",
            "📚 Tentang Model",
            "⚙️ Kelola Standar"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📈 Info Sistem")
    
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
    st.info("🔧 Version: 4.0.0\n📅 Updated: 2025")

# ==========================================
# PAGE 1: DASHBOARD UTAMA
# ==========================================
if page == "🏠 Dashboard Utama":
    
    if df is None:
        st.error("❌ Data tidak tersedia. Jalankan `python main.py` terlebih dahulu untuk memproses data.")
        st.code("python main.py", language="bash")
        st.stop()
    
    st.markdown('<div class="section-header">📊 Ringkasan Cepat</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👨‍🌾 Total Petani", f"{len(df):,}")
    
    with col2:
        if standards_enabled and 'Final_Status' in df.columns:
            underuse = (df['Final_Status'] == 'Underuse').sum()
            st.metric("📉 Underuse", underuse, 
                     delta=f"-{underuse/len(df)*100:.1f}%", 
                     delta_color="inverse")
        else:
            st.metric("📉 Underuse", "N/A", help="Aktifkan standar pupuk")
    
    with col3:
        if standards_enabled and 'Final_Status' in df.columns:
            overuse = (df['Final_Status'] == 'Overuse').sum()
            st.metric("📈 Overuse", overuse,
                     delta=f"+{overuse/len(df)*100:.1f}%",
                     delta_color="inverse")
        else:
            st.metric("📈 Overuse", "N/A", help="Aktifkan standar pupuk")
    
    with col4:
        if 'Cluster_ID' in df.columns:
            clusters = df['Cluster_ID'].nunique()
            st.metric("🎯 Clusters", clusters)
        else:
            st.metric("🎯 Clusters", "N/A")
    
    st.markdown("---")
    
    if standards_enabled and 'Final_Status' in df.columns:
        st.markdown('<div class="section-header">📊 Distribusi Status Penggunaan Pupuk</div>', unsafe_allow_html=True)
        
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
        st.markdown('<div class="section-header">🌱 Perbandingan: Pupuk Aktual vs Standar</div>', unsafe_allow_html=True)
        
        all_standards = standards_manager.get_all_standards()
        
        if all_standards and all(col in df.columns for col in ['Komoditas', 'Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha']):
            komoditas_list = []
            actual_urea = []
            actual_npk = []
            actual_organik = []
            std_urea_mid = []
            std_npk_mid = []
            std_organik_mid = []
            
            for komoditas in all_standards.keys():
                komoditas_data = df[df['Komoditas'] == komoditas]
                if len(komoditas_data) > 0:
                    komoditas_list.append(komoditas)
                    actual_urea.append(komoditas_data['Urea_per_ha'].mean())
                    actual_npk.append(komoditas_data['NPK_per_ha'].mean())
                    actual_organik.append(komoditas_data['Organik_per_ha'].mean())
                    
                    std = all_standards[komoditas]
                    std_urea_mid.append((std['Urea']['min'] + std['Urea']['max']) / 2)
                    std_npk_mid.append((std['NPK']['min'] + std['NPK']['max']) / 2)
                    std_organik_mid.append((std['Organik']['min'] + std['Organik']['max']) / 2)
            
            if komoditas_list:
                tab1, tab2, tab3 = st.tabs(["🌾 Urea", "🌱 NPK", "🍂 Organik"])
                
                with tab1:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=komoditas_list, 
                        y=actual_urea, 
                        name='Aktual', 
                        marker_color='#66b5f6'
                    ))
                    fig.add_trace(go.Bar(
                        x=komoditas_list, 
                        y=std_urea_mid, 
                        name='Standar', 
                        marker_color='#1976d2'
                    ))
                    fig.update_layout(
                        title="Urea: Penggunaan Aktual vs Standar (kg/ha)", 
                        barmode='group', 
                        height=400,
                        xaxis_title="Komoditas",
                        yaxis_title="kg/ha"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=komoditas_list, 
                        y=actual_npk, 
                        name='Aktual', 
                        marker_color='#81c784'
                    ))
                    fig.add_trace(go.Bar(
                        x=komoditas_list, 
                        y=std_npk_mid, 
                        name='Standar', 
                        marker_color='#388e3c'
                    ))
                    fig.update_layout(
                        title="NPK: Penggunaan Aktual vs Standar (kg/ha)", 
                        barmode='group', 
                        height=400,
                        xaxis_title="Komoditas",
                        yaxis_title="kg/ha"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=komoditas_list, 
                        y=actual_organik, 
                        name='Aktual', 
                        marker_color='#ffb74d'
                    ))
                    fig.add_trace(go.Bar(
                        x=komoditas_list, 
                        y=std_organik_mid, 
                        name='Standar', 
                        marker_color='#f57c00'
                    ))
                    fig.update_layout(
                        title="Organik: Penggunaan Aktual vs Standar (kg/ha)", 
                        barmode='group', 
                        height=400,
                        xaxis_title="Komoditas",
                        yaxis_title="kg/ha"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-header">📏 Analisis Luas Lahan & Intensitas Pupuk</div>', unsafe_allow_html=True)
    
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
                title='Distribusi Intensitas Pupuk (kg/ha)',
                color_discrete_sequence=['#4caf50']
            )
            fig.update_layout(xaxis_title="Intensitas Pupuk (kg/ha)", yaxis_title="Jumlah Petani")
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 2: DATA EXPLORER (ENHANCED WITH VALIDATION)
# ==========================================
elif page == "📊 Data Explorer":
    st.markdown('<div class="section-header">📊 Data Explorer</div>', unsafe_allow_html=True)
    
    if df is None:
        st.error("❌ Data tidak tersedia")
        st.stop()
    
    st.markdown("""
    <div class="info-box">
    <strong>💡 Pahami Dulu:</strong><br>
    <ul style="margin: 10px 0; padding-left: 20px;">
        <li><strong>NORMAL</strong> = Mengikuti pola mayoritas petani dalam penggunaan total pupuk per jenis (bukan berarti benar/salah)</li>
        <li><strong>ANOMALI</strong> = Berbeda dari pola umum dalam penggunaan total pupuk per jenis (bukan pelanggaran)</li>
        <li>Fokus analisis: <strong>Total Urea, NPK, Organik per hektar</strong> (tidak berdasarkan MT)</li>
        <li>Anomali bisa lebih baik atau lebih buruk dari normal - yang penting berbeda dari mayoritas</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if 'Desa' in df.columns:
            desa_options = ['Semua'] + sorted(df['Desa'].unique().tolist())
            desa_filter = st.selectbox("🏘️ Desa", desa_options)
        else:
            desa_filter = 'Semua'
    
    with col2:
        if 'Komoditas' in df.columns:
            komoditas_options = ['Semua'] + sorted(df['Komoditas'].unique().tolist())
            komoditas_filter = st.selectbox("🌾 Komoditas", komoditas_options)
        else:
            komoditas_filter = 'Semua'
    
    with col3:
        if standards_enabled and 'Final_Status' in df.columns:
            status_options = ['Semua', 'Normal', 'Underuse', 'Overuse']
            status_filter = st.selectbox("📊 Status Pupuk", status_options)
        else:
            status_filter = 'Semua'
    
    with col4:
        if 'Anomaly_Label' in df.columns:
            anomaly_options = ['Semua', 'Normal', 'Ringan', 'Berat']
            anomaly_filter = st.selectbox("🔍 Anomali", anomaly_options)
        else:
            anomaly_filter = 'Semua'
    
    with col5:
        if 'Prioritas' in df.columns:
            priority_options = ['Semua', 'Tinggi', 'Sedang', 'Rendah']
            priority_filter = st.selectbox("🎯 Prioritas", priority_options)
        else:
            priority_filter = 'Semua'
    
    # Apply filters
    df_filtered = df.copy()
    
    if desa_filter != 'Semua' and 'Desa' in df.columns:
        df_filtered = df_filtered[df_filtered['Desa'] == desa_filter]
    
    if komoditas_filter != 'Semua' and 'Komoditas' in df.columns:
        df_filtered = df_filtered[df_filtered['Komoditas'] == komoditas_filter]
    
    if status_filter != 'Semua' and 'Final_Status' in df.columns:
        df_filtered = df_filtered[df_filtered['Final_Status'] == status_filter]
    
    if anomaly_filter != 'Semua' and 'Anomaly_Label' in df.columns:
        df_filtered = df_filtered[df_filtered['Anomaly_Label'] == anomaly_filter]
    
    if priority_filter != 'Semua' and 'Prioritas' in df.columns:
        df_filtered = df_filtered[df_filtered['Prioritas'] == priority_filter]
    
    st.info(f"📊 Menampilkan {len(df_filtered):,} dari {len(df):,} petani")
    
    display_cols = [
        'ID_Petani', 'Desa', 'Kelompok_Tani', 'Komoditas', 
        'Luas_Tanah_ha', 'Total_Urea', 'Total_NPK', 'Total_Organik',
        'Total_per_ha'
    ]
    
    if standards_enabled and 'Final_Status' in df_filtered.columns:
        display_cols.append('Final_Status')
    
    if 'Anomaly_Label' in df_filtered.columns:
        display_cols.append('Anomaly_Label')
    
    if 'Prioritas' in df_filtered.columns:
        display_cols.append('Prioritas')
    
    display_cols = [col for col in display_cols if col in df_filtered.columns]
    
    # Color highlighting
    def highlight_status(row):
        colors = []
        for col in row.index:
            if col == 'Final_Status':
                if row[col] == 'Overuse':
                    colors.append('background-color: #ffcdd2')
                elif row[col] == 'Underuse':
                    colors.append('background-color: #fff3e0')
                else:
                    colors.append('background-color: #c8e6c9')
            elif col == 'Anomaly_Label':
                if row[col] == 'Berat':
                    colors.append('background-color: #ff8a80; font-weight: bold')
                elif row[col] == 'Ringan':
                    colors.append('background-color: #ffe0b2')
                else:
                    colors.append('')
            elif col == 'Prioritas':
                if row[col] == 'Tinggi':
                    colors.append('background-color: #ff8a80; font-weight: bold')
                elif row[col] == 'Sedang':
                    colors.append('background-color: #ffe0b2')
                else:
                    colors.append('background-color: #c8e6c9')
            else:
                colors.append('')
        return colors
    
    st.dataframe(
        df_filtered[display_cols].style.apply(highlight_status, axis=1),
        use_container_width=True, 
        height=400
    )
    
    st.markdown("---")
    
    st.markdown('<div class="section-header">📈 POLA NORMAL KOMODITAS</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>❓ Yang Disebut "Normal" Itu Seperti Apa?</strong><br>
    Normal = pola yang diikuti oleh mayoritas petani berdasarkan <strong>total penggunaan pupuk per jenis per hektar</strong>. 
    Bukan berarti "benar", tapi ini cara kebanyakan petani menggunakan pupuk untuk komoditas tersebut.
    </div>
    """, unsafe_allow_html=True)
    
    if 'Komoditas' in df.columns:
        selected_komoditas = st.selectbox(
            "Pilih Komoditas untuk Melihat Pola Normal:",
            sorted(df['Komoditas'].unique().tolist())
        )
        
        df_komoditas = df[df['Komoditas'] == selected_komoditas]
        
        # Validate data exists
        if len(df_komoditas) == 0:
            st.warning(f"⚠️ Tidak ada data untuk komoditas {selected_komoditas}")
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Jumlah Petani",
                    f"{len(df_komoditas):,}",
                    help="Jumlah petani yang menanam komoditas ini"
                )
            
            with col2:
                if 'Urea_per_ha' in df_komoditas.columns:
                    median_urea = df_komoditas['Urea_per_ha'].median()
                    st.metric(
                        "Median Urea/ha",
                        f"{median_urea:.1f} kg",
                        help="Nilai tengah TOTAL Urea per hektar (semua musim tanam)"
                    )
            
            with col3:
                if 'NPK_per_ha' in df_komoditas.columns:
                    median_npk = df_komoditas['NPK_per_ha'].median()
                    st.metric(
                        "Median NPK/ha",
                        f"{median_npk:.1f} kg",
                        help="Nilai tengah TOTAL NPK per hektar (semua musim tanam)"
                    )
            
            with col4:
                if 'Organik_per_ha' in df_komoditas.columns:
                    median_organik = df_komoditas['Organik_per_ha'].median()
                    st.metric(
                        "Median Organik/ha",
                        f"{median_organik:.1f} kg",
                        help="Nilai tengah TOTAL pupuk Organik per hektar (semua musim tanam)"
                    )
            
            st.markdown("#### 📊 Distribusi Total Penggunaan Pupuk per Hektar")
            st.markdown("*Menampilkan total penggunaan pupuk per jenis (bukan per MT)*")
            
            has_data = False
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Total Urea (kg/ha)', 'Total NPK (kg/ha)', 'Total Organik (kg/ha)')
            )
            
            col_idx = 1
            for pupuk, color in [('Urea', '#2196F3'), ('NPK', '#4CAF50'), ('Organik', '#FF9800')]:
                col_name = f'{pupuk}_per_ha'
                if col_name in df_komoditas.columns:
                    data = df_komoditas[col_name].dropna()
                    if len(data) > 0:
                        has_data = True
                        
                        # Add boxplot
                        fig.add_trace(
                            go.Box(
                                y=data,
                                name=pupuk,
                                marker_color=color,
                                boxmean='sd'
                            ),
                            row=1, col=col_idx
                        )
                col_idx += 1
            
            if has_data:
                fig.update_layout(
                    title_text=f"Distribusi Total Pupuk per Hektar - {selected_komoditas}",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div class="success-box">
                <strong>💡 Cara Membaca:</strong> Kotak menunjukkan rentang normal (50% petani di tengah). 
                Garis tengah = median (nilai tengah). Titik di luar kotak = outlier (berbeda dari mayoritas).
                <br><br>
                <strong>⚠️ Penting:</strong> Ini adalah TOTAL penggunaan pupuk per jenis (bukan per MT). 
                Petani dengan total jauh dari median akan terdeteksi sebagai anomali.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Data pupuk per hektar tidak tersedia untuk visualisasi ini.")
    
    st.markdown("---")
    
    st.markdown('<div class="section-header">🎯 POSISI PETANI TERHADAP POLA NORMAL</div>', unsafe_allow_html=True)
    
    selected_id = st.selectbox(
        "Pilih ID Petani untuk Analisis Detail:",
        ['-- Pilih --'] + df_filtered['ID_Petani'].tolist()
    )
    
    if selected_id != '-- Pilih --':
        petani_data = df_filtered[df_filtered['ID_Petani'] == selected_id].iloc[0]
        komoditas_petani = petani_data.get('Komoditas', 'Unknown')
        
        # Validate commodity exists
        df_same_commodity = df[df['Komoditas'] == komoditas_petani]
        
        if len(df_same_commodity) < 2:
            st.warning(f"⚠️ Tidak cukup data untuk perbandingan komoditas {komoditas_petani}")
        else:
            st.markdown(f"### 📋 Detail: {selected_id} ({komoditas_petani})")
            
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
                st.markdown("**📊 Pupuk per Hektar**")
                if 'Urea_per_ha' in petani_data.index:
                    st.write(f"Urea: {petani_data.get('Urea_per_ha', 0):.1f} kg/ha")
                if 'NPK_per_ha' in petani_data.index:
                    st.write(f"NPK: {petani_data.get('NPK_per_ha', 0):.1f} kg/ha")
                if 'Organik_per_ha' in petani_data.index:
                    st.write(f"Organik: {petani_data.get('Organik_per_ha', 0):.1f} kg/ha")
            
            st.markdown("---")
            
            st.markdown("#### 📊 Perbandingan dengan Median Komoditas")
            st.markdown("*Membandingkan total penggunaan pupuk per hektar dengan mayoritas petani*")
            
            # Calculate median for commodity
            pupuk_types = ['Urea', 'NPK', 'Organik']
            petani_values = []
            median_values = []
            labels = []
            
            for pupuk in pupuk_types:
                per_ha_col = f'{pupuk}_per_ha'
                if per_ha_col in petani_data.index and per_ha_col in df_same_commodity.columns:
                    petani_val = petani_data.get(per_ha_col, 0)
                    median_val = df_same_commodity[per_ha_col].median()
                    
                    petani_values.append(petani_val)
                    median_values.append(median_val)
                    labels.append(f'{pupuk}\n(kg/ha)')
            
            if petani_values:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=labels,
                    y=median_values,
                    name=f'Median {komoditas_petani}',
                    marker_color='#90CAF9',
                    text=[f'{v:.1f}' for v in median_values],
                    textposition='auto'
                ))
                
                fig.add_trace(go.Bar(
                    x=labels,
                    y=petani_values,
                    name=f'Petani {selected_id}',
                    marker_color='#EF5350',
                    text=[f'{v:.1f}' for v in petani_values],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title=f'Petani vs Median - Total per Hektar',
                    barmode='group',
                    height=400,
                    yaxis_title='Total Pupuk per Hektar (kg/ha)',
                    xaxis_title='Jenis Pupuk'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div class="info-box">
                <strong>💡 Interpretasi:</strong> Grafik ini menunjukkan posisi petani dibandingkan dengan 
                pola umum (median) untuk <strong>total penggunaan pupuk per jenis</strong>. 
                <br><br>
                Jika nilai petani jauh berbeda dari median (baik lebih tinggi atau lebih rendah), 
                maka akan terdeteksi sebagai anomali. <strong>Fokus pada total per jenis, bukan distribusi per MT.</strong>
                </div>
                """, unsafe_allow_html=True)
            
            # Show anomaly status
            if 'Anomaly_Label' in petani_data.index:
                anomaly_label = petani_data.get('Anomaly_Label', 'Normal')
                anomaly_score = petani_data.get('Anomaly_Score', 0.0)
                
                if anomaly_label == 'Anomali':
                    st.error(f"⚠️ Status: **ANOMALI** (Skor: {anomaly_score:.3f})")
                else:
                    st.success(f"✅ Status: **NORMAL** (Skor: {anomaly_score:.3f})")
                
                # Get explanation
                try:
                    from src.anomaly_explain import get_anomaly_explanation
                    explanation = get_anomaly_explanation(selected_id, df)
                    
                    st.markdown("#### 💡 Penjelasan Anomali")
                    st.markdown(explanation['explanation_text'])
                except Exception as e:
                    st.warning(f"Tidak dapat memuat penjelasan detail: {e}")
            
            st.markdown("---")
            
            st.markdown("#### 📊 VISUALISASI 2: Total Pupuk per Ha (Petani vs Median)")
            
            comparison_data = []
            pupuk_types = []
            petani_values = []
            median_values = []
            
            for pupuk in ['Urea', 'NPK', 'Organik']:
                col_per_ha = f'{pupuk}_per_ha' # Use '_per_ha' for comparison
                if col_per_ha in petani_data.index and col_per_ha in df_same_commodity.columns:
                    petani_val = petani_data[col_per_ha]
                    median_val = df_same_commodity[col_per_ha].median()
                    
                    # Validate values
                    if pd.notna(petani_val) and pd.notna(median_val):
                        pupuk_types.append(f'{pupuk}/ha')
                        petani_values.append(petani_val)
                        median_values.append(median_val)
                        
                        diff_pct = ((petani_val - median_val) / median_val * 100) if median_val > 0 else 0
                        comparison_data.append({
                            'Jenis Pupuk': f'{pupuk} per ha',
                            'Petani': f"{petani_val:.1f} kg",
                            'Median': f"{median_val:.1f} kg",
                            'Selisih': f"{diff_pct:+.1f}%"
                        })
            
            if comparison_data:
                # Show table
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Show chart
                fig_compare = go.Figure()
                
                fig_compare.add_trace(go.Bar(
                    x=pupuk_types,
                    y=petani_values,
                    name=f'Petani {selected_id}',
                    marker_color='#FF5722',
                    text=[f'{v:.1f}' for v in petani_values],
                    textposition='auto'
                ))
                
                fig_compare.add_trace(go.Bar(
                    x=pupuk_types,
                    y=median_values,
                    name=f'Median {komoditas_petani}',
                    marker_color='#4CAF50',
                    text=[f'{v:.1f}' for v in median_values],
                    textposition='auto'
                ))
                
                fig_compare.update_layout(
                    title=f"Perbandingan Penggunaan Pupuk: Petani {selected_id} vs Median {komoditas_petani}",
                    yaxis_title="kg/ha",
                    barmode='group',
                    height=450
                )
                
                st.plotly_chart(fig_compare, use_container_width=True)
                
                # Generate contextual interpretation
                interpretations = []
                for i, data in enumerate(comparison_data):
                    selisih_str = data['Selisih'].replace('%', '').replace('+', '')
                    try:
                        selisih_pct = float(selisih_str)
                        pupuk_name = data['Jenis Pupuk'].split(' per')[0]
                        
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
                    except:
                        pass
                
                if interpretations:
                    st.markdown("""
                    <div class="info-box">
                    <strong>💡 Interpretasi Grafik:</strong><br>
                    {}
                    </div>
                    """.format('<br>'.join(interpretations)), unsafe_allow_html=True)
            else:
                st.warning("⚠️ Data pupuk per hektar tidak tersedia untuk perbandingan.")
            
            st.markdown("---")
            
            st.markdown("#### 📊 VISUALISASI 3: Distribusi Pupuk Sepanjang Musim Tanam (MT1-MT3)")
            
            mt_data = []
            
            for pupuk_type in ['Urea', 'NPK', 'Organik']:
                mt1_col = f'{pupuk_type}_MT1'
                mt2_col = f'{pupuk_type}_MT2'
                mt3_col = f'{pupuk_type}_MT3'
                
                if all(col in petani_data.index for col in [mt1_col, mt2_col, mt3_col]):
                    mt1_val = petani_data[mt1_col]
                    mt2_val = petani_data[mt2_col]
                    mt3_val = petani_data[mt3_col]
                    
                    # Validate values
                    if pd.notna(mt1_val) and pd.notna(mt2_val) and pd.notna(mt3_val):
                        mt_data.append({
                            'Pupuk': pupuk_type,
                            'MT1': mt1_val,
                            'MT2': mt2_val,
                            'MT3': mt3_val
                        })
            
            if mt_data:
                mt_df = pd.DataFrame(mt_data)
                
                # Show table first
                st.dataframe(mt_df, use_container_width=True, hide_index=True)
                
                # Stacked bar chart
                fig_mt = go.Figure()
                
                fig_mt.add_trace(go.Bar(
                    name='MT1 (Musim Tanam 1)',
                    x=mt_df['Pupuk'],
                    y=mt_df['MT1'],
                    marker_color='#42A5F5',
                    text=mt_df['MT1'].apply(lambda x: f'{x:.1f}'),
                    textposition='inside'
                ))
                
                fig_mt.add_trace(go.Bar(
                    name='MT2 (Musim Tanam 2)',
                    x=mt_df['Pupuk'],
                    y=mt_df['MT2'],
                    marker_color='#66BB6A',
                    text=mt_df['MT2'].apply(lambda x: f'{x:.1f}'),
                    textposition='inside'
                ))
                
                fig_mt.add_trace(go.Bar(
                    name='MT3 (Musim Tanam 3)',
                    x=mt_df['Pupuk'],
                    y=mt_df['MT3'],
                    marker_color='#FFA726',
                    text=mt_df['MT3'].apply(lambda x: f'{x:.1f}'),
                    textposition='inside'
                ))
                
                fig_mt.update_layout(
                    title=f"Distribusi Penggunaan Pupuk Sepanjang Musim Tanam - Petani {selected_id}",
                    yaxis_title="kg",
                    barmode='stack',
                    height=450,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_mt, use_container_width=True)
                
                # Generate MT interpretation
                total_by_mt = {
                    'MT1': sum([d['MT1'] for d in mt_data]),
                    'MT2': sum([d['MT2'] for d in mt_data]),
                    'MT3': sum([d['MT3'] for d in mt_data])
                }
                
                dominant_mt = max(total_by_mt, key=total_by_mt.get)
                weakest_mt = min(total_by_mt, key=total_by_mt.get)
                
                # Check balance
                max_mt_val = total_by_mt[dominant_mt]
                min_mt_val = total_by_mt[weakest_mt]
                
                if max_mt_val > 0 and min_mt_val == 0:
                    balance_status = "TIDAK SEIMBANG - Ada musim tanam yang tidak diberi pupuk sama sekali"
                    balance_emoji = "🔴"
                elif max_mt_val > 0 and (min_mt_val == 0 or (min_mt_val > 0 and (max_mt_val / min_mt_val) > 5)):
                    balance_status = "KURANG SEIMBANG - Satu musim tanam mendominasi"
                    balance_emoji = "🟡"
                else:
                    balance_status = "CUKUP SEIMBANG - Distribusi relatif merata"
                    balance_emoji = "✅"
                
                st.markdown(f"""
                <div class="info-box">
                <strong>💡 Interpretasi Pola Musim Tanam:</strong><br>
                • <strong>{dominant_mt}</strong> adalah periode dengan penggunaan pupuk TERTINGGI: {total_by_mt[dominant_mt]:.1f} kg total<br>
                • <strong>{weakest_mt}</strong> adalah periode dengan penggunaan pupuk TERENDAH: {total_by_mt[weakest_mt]:.1f} kg total<br>
                • {balance_emoji} Status Distribusi: <strong>{balance_status}</strong><br>
                <br>
                <em>Catatan:</em> Distribusi yang seimbang biasanya menunjukkan perawatan tanaman yang konsisten. 
                Namun, beberapa komoditas memang memerlukan pemupukan intensif pada musim tertentu.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Data distribusi musim tanam (MT1-MT3) tidak tersedia.")
            
            st.markdown("---")
            
            st.markdown("#### 📊 VISUALISASI 4: Proporsi Jenis Pupuk (%)")
            
            total_urea = petani_data.get('Total_Urea', 0)
            total_npk = petani_data.get('Total_NPK', 0)
            total_organik = petani_data.get('Total_Organik', 0)
            total_all = total_urea + total_npk + total_organik
            
            if total_all > 0 and pd.notna(total_all):
                prop_urea = (total_urea / total_all) * 100
                prop_npk = (total_npk / total_all) * 100
                prop_organik = (total_organik / total_all) * 100
                
                # Calculate median proportions for commodity
                df_commodity_valid = df_same_commodity.copy()
                df_commodity_valid['total_pupuk'] = (
                    df_commodity_valid.get('Total_Urea', 0) + 
                    df_commodity_valid.get('Total_NPK', 0) + 
                    df_commodity_valid.get('Total_Organik', 0)
                )
                
                df_commodity_valid = df_commodity_valid[df_commodity_valid['total_pupuk'] > 0]
                
                if len(df_commodity_valid) > 0:
                    df_commodity_valid['Prop_Urea'] = (df_commodity_valid.get('Total_Urea', 0) / 
                                                       df_commodity_valid['total_pupuk']) * 100
                    df_commodity_valid['Prop_NPK'] = (df_commodity_valid.get('Total_NPK', 0) / 
                                                      df_commodity_valid['total_pupuk']) * 100
                    df_commodity_valid['Prop_Organik'] = (df_commodity_valid.get('Total_Organik', 0) / 
                                                          df_commodity_valid['total_pupuk']) * 100
                    
                    median_prop_urea = df_commodity_valid['Prop_Urea'].median()
                    median_prop_npk = df_commodity_valid['Prop_NPK'].median()
                    median_prop_organik = df_commodity_valid['Prop_Organik'].median()
                    
                    categories = ['Urea', 'NPK', 'Organik']
                    petani_props = [prop_urea, prop_npk, prop_organik]
                    median_props = [median_prop_urea, median_prop_npk, median_prop_organik]
                    
                    # Show table
                    prop_table = pd.DataFrame({
                        'Jenis Pupuk': categories,
                        f'Petani {selected_id} (%)': [f'{p:.1f}%' for p in petani_props],
                        f'Median {komoditas_petani} (%)': [f'{p:.1f}%' for p in median_props]
                    })
                    st.dataframe(prop_table, use_container_width=True, hide_index=True)
                    
                    # Bar chart
                    fig_prop = go.Figure()
                    
                    fig_prop.add_trace(go.Bar(
                        x=categories,
                        y=petani_props,
                        name=f'Petani {selected_id}',
                        marker_color='#FF5722',
                        text=[f'{p:.1f}%' for p in petani_props],
                        textposition='auto'
                    ))
                    
                    fig_prop.add_trace(go.Bar(
                        x=categories,
                        y=median_props,
                        name=f'Median {komoditas_petani}',
                        marker_color='#4CAF50',
                        text=[f'{p:.1f}%' for p in median_props],
                        textposition='auto'
                    ))
                    
                    fig_prop.update_layout(
                        title="Komposisi Jenis Pupuk (%)",
                        yaxis_title="Persentase (%)",
                        barmode='group',
                        height=450
                    )
                    
                    st.plotly_chart(fig_prop, use_container_width=True)
                    
                    # Interpretation with specific insights
                    prop_interpretations = []
                    for i, pupuk in enumerate(categories):
                        petani_p = petani_props[i]
                        median_p = median_props[i]
                        diff = petani_p - median_p
                        
                        if abs(diff) < 5:
                            prop_interpretations.append(f"✅ {pupuk}: Proporsi sesuai median ({petani_p:.1f}% vs {median_p:.1f}%)")
                        elif diff > 15:
                            prop_interpretations.append(f"🔴 {pupuk}: Proporsi JAUH LEBIH TINGGI ({petani_p:.1f}% vs {median_p:.1f}%, selisih +{diff:.1f}%)")
                        elif diff < -15:
                            prop_interpretations.append(f"🟡 {pupuk}: Proporsi JAUH LEBIH RENDAH ({petani_p:.1f}% vs {median_p:.1f}%, selisih {diff:.1f}%)")
                        elif diff > 0:
                            prop_interpretations.append(f"🟠 {pupuk}: Sedikit lebih tinggi ({petani_p:.1f}% vs {median_p:.1f}%)")
                        else:
                            prop_interpretations.append(f"🟠 {pupuk}: Sedikit lebih rendah ({petani_p:.1f}% vs {median_p:.1f}%)")
                    
                    st.markdown(f"""
                    <div class="info-box">
                    <strong>💡 Interpretasi Komposisi Pupuk:</strong><br>
                    {('<br>'.join(prop_interpretations))}<br>
                    <br>
                    <em>Total pupuk petani ini: {total_all:.1f} kg ({prop_urea:.1f}% Urea, {prop_npk:.1f}% NPK, {prop_organik:.1f}% Organik)</em>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Tidak cukup data komoditas untuk perbandingan proporsi.")
            else:
                st.warning("⚠️ Total pupuk petani ini adalah 0, tidak dapat menghitung proporsi.")
            
            st.markdown("---")
            
            st.markdown("#### 📊 VISUALISASI 5: TINGKAT ANOMALI & PENJELASAN")
            
            st.markdown("""
            <div class="info-box">
            <strong>📚 Ingat:</strong> Anomali ≠ Kesalahan. Anomali hanya berarti "berbeda dari pola mayoritas". 
            Bisa jadi petani ini lebih baik, atau memiliki kondisi lahan yang unik.
            </div>
            """, unsafe_allow_html=True)
            
            anomaly_label = petani_data.get('Anomaly_Label', 'Unknown')
            anomaly_score = petani_data.get('anomaly_score', petani_data.get('Anomaly_Score', 0)) # Fallback for different naming conventions
            
            # Create visualization of anomaly level
            col1, col2, col3 = st.columns([1,2,1])
            
            with col2:
                if anomaly_label == 'Normal':
                    st.markdown("""
                    <div style="background-color: #4CAF50; color: white; padding: 30px; border-radius: 10px; text-align: center;">
                        <h2 style="margin: 0;">✅ NORMAL</h2>
                        <p style="margin: 10px 0 0 0; font-size: 18px;">Pola Mengikuti Mayoritas</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif anomaly_label == 'Ringan':
                    st.markdown("""
                    <div style="background-color: #FF9800; color: white; padding: 30px; border-radius: 10px; text-align: center;">
                        <h2 style="margin: 0;">⚠️ ANOMALI RINGAN</h2>
                        <p style="margin: 10px 0 0 0; font-size: 18px;">Sedikit Berbeda dari Mayoritas</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:  # Berat
                    st.markdown("""
                    <div style="background-color: #F44336; color: white; padding: 30px; border-radius: 10px; text-align: center;">
                        <h2 style="margin: 0;">🚨 ANOMALI BERAT</h2>
                        <p style="margin: 10px 0 0 0; font-size: 18px;">Sangat Berbeda dari Mayoritas</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Show anomaly score if available
            if pd.notna(anomaly_score):
                st.metric("Skor Anomali (dari model ML)", f"{anomaly_score:.4f}", 
                         help="Skor dari Isolation Forest. Nilai negatif = anomali, semakin negatif semakin anomali.")
            
            # SPECIFIC EXPLANATION WITH EVIDENCE
            if anomaly_label == 'Normal':
                st.success(f"✅ Petani {selected_id} memiliki pola penggunaan pupuk yang NORMAL")
                st.markdown("""
                <strong>Artinya:</strong> Pola penggunaan pupuk petani ini serupa dengan kebanyakan petani lain yang menanam komoditas yang sama.
                Tidak ada yang mencurigakan atau sangat berbeda dari pola umum.
                """)
            else:
                anomaly_factors = []
                
                # SPECIFIC FACTOR 1: Check extreme per ha values
                for pupuk in ['Urea', 'NPK', 'Organik']:
                    col_per_ha = f'{pupuk}_per_ha' # Use '_per_ha' for comparison
                    if col_per_ha in petani_data.index and col_per_ha in df_same_commodity.columns:
                        petani_val = petani_data[col_per_ha]
                        median_val = df_same_commodity[col_per_ha].median()
                        
                        if pd.notna(petani_val) and pd.notna(median_val) and median_val > 0:
                            diff_pct = ((petani_val - median_val) / median_val) * 100
                            
                            if abs(diff_pct) > 50:
                                anomaly_factors.append(
                                    f"<strong>{pupuk} per ha sangat tinggi</strong>: Petani menggunakan {petani_val:.1f} kg/ha, "
                                    f"sedangkan median komoditas ini hanya {median_val:.1f} kg/ha. "
                                    f"Selisih {abs(diff_pct):.1f}% lebih tinggi dari pola umum. "
                                    f"(lihat grafik Visualisasi 2 di atas)"
                                )
                
                # SPECIFIC FACTOR 2: Check MT distribution imbalance
                if mt_data and len(mt_data) > 0:
                    for pupuk_mt in mt_data:
                        pupuk_name = pupuk_mt['Pupuk']
                        mt_values = [pupuk_mt['MT1'], pupuk_mt['MT2'], pupuk_mt['MT3']]
                        max_mt_val = max(mt_values)
                        min_mt_val = min(mt_values)
                        
                        if max_mt_val > 0:
                            if min_mt_val == 0 and max_mt_val > 50:
                                anomaly_factors.append(
                                    f"**Distribusi {pupuk_name} tidak seimbang**: Ada musim tanam yang tidak diberi {pupuk_name} "
                                    f"(0 kg), sementara musim lain mendapat {max_mt_val:.1f} kg. "
                                    f"Pola ini jarang ditemukan. (lihat grafik Visualisasi 3 di atas)"
                                )
                            elif min_mt_val > 0 and (max_mt_val / min_mt_val) > 10:
                                anomaly_factors.append(
                                    f"**Distribusi {pupuk_name} sangat timpang**: Musim tanam tertinggi ({max_mt_val:.1f} kg) "
                                    f"lebih dari 10x lebih besar dari terendah ({min_mt_val:.1f} kg). "
                                    f"(lihat grafik Visualisasi 3 di atas)"
                                )
                
                # SPECIFIC FACTOR 3: Check proportion imbalance
                if total_all > 0:
                    if prop_urea > 80:
                        anomaly_factors.append(
                            f"**Proporsi Urea terlalu dominan**: {prop_urea:.1f}% dari total pupuk adalah Urea. "
                            f"Pola ini jarang - kebanyakan petani menggunakan campuran lebih seimbang. "
                            f"(lihat grafik Visualisasi 4 di atas)"
                        )
                    elif prop_organik > 70:
                        anomaly_factors.append(
                            f"**Proporsi Organik sangat tinggi**: {prop_organik:.1f}% dari total pupuk adalah Organik. "
                            f"Ini tidak umum untuk komoditas {komoditas_petani}. "
                            f"(lihat grafik Visualisasi 4 di atas)"
                        )
                
                # SPECIFIC FACTOR 4: Check luas lahan
                luas_ha = petani_data.get('Luas_Tanah_ha', 0)
                if 'Luas_Tanah_ha' in df_same_commodity.columns:
                    median_luas = df_same_commodity['Luas_Tanah_ha'].median()
                    
                    if luas_ha > 0 and median_luas > 0:
                        luas_diff_pct = ((luas_ha - median_luas) / median_luas) * 100
                        if abs(luas_diff_pct) > 100:
                            if luas_diff_pct > 0:
                                anomaly_factors.append(
                                    f"**Luas lahan sangat besar**: Petani memiliki {luas_ha:.2f} ha, "
                                    f"{abs(luas_diff_pct):.1f}% lebih besar dari median ({median_luas:.2f} ha). "
                                    f"Lahan yang sangat besar bisa menghasilkan pola pupuk yang berbeda."
                                )
                            else:
                                anomaly_factors.append(
                                    f"**Luas lahan sangat kecil**: Petani memiliki {luas_ha:.2f} ha, "
                                    f"{abs(luas_diff_pct):.1f}% lebih kecil dari median ({median_luas:.2f} ha). "
                                    f"Lahan yang sangat kecil bisa menghasilkan pola pupuk yang unik."
                                )
                
                # Display findings
                if anomaly_label == 'Berat':
                    st.error(f"🚨 Petani {selected_id} terdeteksi sebagai ANOMALI BERAT")
                else:
                    st.warning(f"⚠️ Petani {selected_id} terdeteksi sebagai ANOMALI RINGAN")
                
                st.markdown("**Ringkasan:** Pola penggunaan pupuk petani ini berbeda signifikan dari mayoritas petani lain.")
                
                if anomaly_factors:
                    st.markdown("**Faktor Spesifik yang Menyebabkan Status Anomali:**")
                    for i, factor in enumerate(anomaly_factors[:3], 1):
                        st.markdown(f"{i}. {factor}")
                else:
                    st.markdown("**Faktor Utama:**")
                    st.markdown(
                        "- Kombinasi berbagai fitur pupuk (dosis, proporsi, distribusi MT) yang jarang ditemukan pada petani lain. "
                        "Meskipun tidak ada satu faktor yang ekstrem, kombinasi keseluruhan menciptakan pola unik yang berbeda dari mayoritas."
                    )
                
                st.markdown("""
                <div class="warning-box">
                <strong>⚠️ Perlu Evaluasi Lebih Lanjut:</strong><br>
                Anomali tidak selalu berarti buruk. Bisa jadi:
                <ul>
                    <li>✅ Kondisi lahan memang berbeda (tanah lebih subur/kurang subur)</li>
                    <li>✅ Petani menggunakan teknik khusus yang lebih efisien</li>
                    <li>⚠️ Ada kesalahan pencatatan data yang perlu diperbaiki</li>
                    <li>⚠️ Praktik yang perlu dievaluasi dan disesuaikan</li>
                </ul>
                <strong>Rekomendasi:</strong> Lakukan verifikasi lapangan untuk memastikan penyebab pola anomali ini.
                </div>
                """, unsafe_allow_html=True)
    
        st.markdown("---")
        
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Data (CSV)",
            data=csv,
            file_name=f"rdkk_filtered_data.csv",
            mime="text/csv",
        )
        
        st.markdown("---")
        st.markdown("### 📈 Visualisasi Data")
        
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "📊 Distribusi Status", 
            "🌾 Per Komoditas", 
            "📍 Per Desa",
            "📦 Boxplot Pupuk"
        ])
        
        with viz_tab1:
            if standards_enabled and 'Final_Status' in df_filtered.columns:
                st.markdown("""
                <strong>Penjelasan:</strong> Grafik ini menunjukkan distribusi status penggunaan pupuk. 
                - <strong>Normal</strong>: Penggunaan sesuai standar
                - <strong>Underuse</strong>: Penggunaan di bawah standar (mungkin hasil kurang optimal)
                - <strong>Overuse</strong>: Penggunaan melebihi standar (pemborosan subsidi)
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    status_counts = df_filtered['Final_Status'].value_counts()
                    fig = px.pie(
                        values=status_counts.values,
                        names=status_counts.index,
                        title='Distribusi Status Pupuk',
                        color=status_counts.index,
                        color_discrete_map={'Normal': '#4caf50', 'Underuse': '#ff9800', 'Overuse': '#f44336'},
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    status_bar = df_filtered['Final_Status'].value_counts().reset_index()
                    status_bar.columns = ['Status', 'Jumlah']
                    fig = px.bar(
                        status_bar,
                        x='Status',
                        y='Jumlah',
                        title='Jumlah Petani per Status',
                        color='Status',
                        color_discrete_map={'Normal': '#4caf50', 'Underuse': '#ff9800', 'Overuse': '#f44336'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab2:
            st.markdown("""
            <strong>Penjelasan:</strong> Analisis penggunaan pupuk berdasarkan jenis komoditas. 
            Setiap komoditas memiliki kebutuhan pupuk berbeda.
            """)
            
            if 'Komoditas' in df_filtered.columns and 'Total_per_ha' in df_filtered.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    pupuk_by_komoditas = df_filtered.groupby('Komoditas')['Total_per_ha'].mean().reset_index()
                    fig = px.bar(
                        pupuk_by_komoditas,
                        x='Komoditas',
                        y='Total_per_ha',
                        title='Rata-rata Intensitas Pupuk per Komoditas (kg/ha)',
                        color='Total_per_ha',
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if standards_enabled and 'Final_Status' in df_filtered.columns:
                        status_by_commodity = df_filtered.groupby(['Komoditas', 'Final_Status']).size().reset_index(name='Jumlah')
                        fig = px.bar(
                            status_by_commodity,
                            x='Komoditas',
                            y='Jumlah',
                            color='Final_Status',
                            title='Status per Komoditas',
                            color_discrete_map={'Normal': '#4caf50', 'Underuse': '#ff9800', 'Overuse': '#f44336'},
                            barmode='stack'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab3:
            if 'Desa' in df_filtered.columns:
                st.markdown("""
                <strong>Penjelasan:</strong> Distribusi petani dan penggunaan pupuk per desa. 
                Membantu identifikasi desa yang perlu perhatian khusus.
                """)
                
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
            st.markdown("""
            <strong>Penjelasan:</strong> Boxplot menunjukkan distribusi, median, dan outlier (nilai ekstrem) penggunaan pupuk.
            - <strong>Kotak</strong>: 50% data tengah
            - <strong>Garis tengah</strong>: Median (nilai tengah)
            - <strong>Titik di luar</strong>: Outlier (anomali potensial)
            """)
            
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

# ==========================================
# PAGE 3: PREDIKSI DATA BARU
# ==========================================
elif page == "🔍 Prediksi Data Baru":
    st.markdown('<div class="section-header">🔍 Prediksi Data Baru</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>ℹ️ Cara Menggunakan:</strong><br>
    1. Masukkan data petani baru di form di bawah<br>
    2. Sistem akan otomatis menghitung status (jika standar aktif)<br>
    3. Rekomendasi akan diberikan berdasarkan analisis ML dan standar (jika aktif)
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        st.subheader("📝 Data Petani Baru")
        
        col1, col2 = st.columns(2)
        
        with col1:
            id_petani = st.text_input("ID Petani", value="P_NEW_001", help="ID unik petani")
            desa = st.text_input("Desa", value="Desa Baru")
            kelompok = st.text_input("Kelompok Tani", value="Kelompok Baru")
        
        with col2:
            all_standards = standards_manager.get_all_standards()
            commodity_options = list(all_standards.keys()) if all_standards else ['PADI', 'JAGUNG', 'KEDELAI', 'KOPI', 'CABAI']
            komoditas = st.selectbox("Komoditas", commodity_options)
            luas_m2 = st.number_input("Luas Tanah (m²)", min_value=100, max_value=100000, value=5000, step=100)
        
        st.markdown("---")
        st.subheader("🌾 Data Pupuk per Musim Tanam (kg)")
        
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
        
        submitted = st.form_submit_button("🔍 Analisis Data", use_container_width=True, type="primary")
    
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
                    st.markdown("### 🎯 Status Keseluruhan")
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
                st.markdown("### 📊 Detail Penggunaan Pupuk")
                
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
# PAGE 4: CLUSTERING & POLA
# ==========================================
elif page == "🎯 Clustering & Pola":
    st.markdown('<div class="section-header">🎯 Clustering & Analisis Pola</div>', unsafe_allow_html=True)
    
    if df is None:
        st.error("❌ Data tidak tersedia")
        st.stop()
    
    if 'Cluster_ID' not in df.columns:
        st.warning("⚠️ Data clustering belum tersedia. Jalankan `python main.py` terlebih dahulu.")
        st.stop()
    
    st.markdown("""
    <div class="info-box">
    <strong>ℹ️ Tentang Clustering:</strong><br>
    Clustering mengelompokkan petani berdasarkan pola penggunaan pupuk yang mirip.
    Ini membantu mengidentifikasi pola-pola tertentu dalam distribusi pupuk subsidi.
    </div>
    """, unsafe_allow_html=True)
    
    # Cluster summary
    n_clusters = df['Cluster_ID'].nunique()
    st.subheader(f"📊 Ringkasan: {n_clusters} Cluster Teridentifikasi")
    
    # Cluster distribution
    cluster_counts = df['Cluster_ID'].value_counts().sort_index()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📈 Distribusi Cluster")
        for cluster_id in sorted(cluster_counts.index):
            count = cluster_counts[cluster_id]
            pct = count / len(df) * 100
            st.metric(f"Cluster {cluster_id}", f"{count} petani", f"{pct:.1f}%")
    
    with col2:
        fig = px.pie(
            values=cluster_counts.values,
            names=[f"Cluster {i}" for i in cluster_counts.index],
            title='Distribusi Petani per Cluster',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster characteristics
    st.markdown("---")
    st.markdown("### 🔍 Karakteristik Setiap Cluster")
    
    for cluster_id in sorted(df['Cluster_ID'].unique()):
        cluster_data = df[df['Cluster_ID'] == cluster_id]
        
        with st.expander(f"📍 Cluster {cluster_id} ({len(cluster_data)} petani)", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_luas = cluster_data['Luas_Tanah_ha'].mean()
                st.metric("Rata-rata Luas", f"{avg_luas:.2f} ha")
            
            with col2:
                if 'Total_Pupuk' in cluster_data.columns:
                    avg_total = cluster_data['Total_Pupuk'].mean()
                    st.metric("Rata-rata Total Pupuk", f"{avg_total:.1f} kg")
            
            with col3:
                if 'Total_per_ha' in cluster_data.columns:
                    avg_intensity = cluster_data['Total_per_ha'].mean()
                    st.metric("Intensitas", f"{avg_intensity:.1f} kg/ha")
            
            with col4:
                if standards_enabled and 'Final_Status' in cluster_data.columns:
                    overuse_pct = (cluster_data['Final_Status'] == 'Overuse').mean() * 100
                    underuse_pct = (cluster_data['Final_Status'] == 'Underuse').mean() * 100
                    
                    if overuse_pct > 30:
                        st.metric("Status Dominan", "Overuse", f"{overuse_pct:.0f}%")
                    elif underuse_pct > 30:
                        st.metric("Status Dominan", "Underuse", f"{underuse_pct:.0f}%")
                    else:
                        st.metric("Status Dominan", "Normal", "Stabil")
            
            # Visualisasi cluster
            if all(col in cluster_data.columns for col in ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha']):
                avg_urea = cluster_data['Urea_per_ha'].mean()
                avg_npk = cluster_data['NPK_per_ha'].mean()
                avg_organik = cluster_data['Organik_per_ha'].mean()
                
                fig = go.Figure(data=[
                    go.Bar(name='Urea', x=['Urea'], y=[avg_urea], marker_color='#64b5f6'),
                    go.Bar(name='NPK', x=['NPK'], y=[avg_npk], marker_color='#81c784'),
                    go.Bar(name='Organik', x=['Organik'], y=[avg_organik], marker_color='#ffb74d')
                ])
                fig.update_layout(
                    title=f"Rata-rata Penggunaan Pupuk (kg/ha) - Cluster {cluster_id}",
                    showlegend=False,
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot: Luas vs Intensitas colored by cluster
    st.markdown("---")
    st.markdown("### 📊 Visualisasi: Luas Lahan vs Intensitas Pupuk")
    
    if all(col in df.columns for col in ['Luas_Tanah_ha', 'Total_per_ha', 'Cluster_ID']):
        fig = px.scatter(
            df,
            x='Luas_Tanah_ha',
            y='Total_per_ha',
            color='Cluster_ID',
            title='Pola Luas Lahan vs Intensitas Pupuk per Cluster',
            labels={'Luas_Tanah_ha': 'Luas Lahan (ha)', 'Total_per_ha': 'Intensitas Pupuk (kg/ha)'},
            hover_data=['Komoditas', 'Total_Pupuk']
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 5: REKOMENDASI (ENHANCED)
# ==========================================
elif page == "💡 Rekomendasi":
    st.markdown('<div class="section-header">💡 Rekomendasi & Action Plan</div>', unsafe_allow_html=True)
    
    if df is None:
        st.error("❌ Data tidak tersedia")
        st.stop()
    
    if 'Rekomendasi' not in df.columns or 'Prioritas' not in df.columns:
        st.warning("⚠️ Rekomendasi belum tersedia. Jalankan `python main.py` terlebih dahulu.")
        st.stop()
    
    st.markdown("""
    <div class="info-box">
    <strong>ℹ️ Tentang Rekomendasi:</strong><br>
    Rekomendasi dihasilkan berdasarkan:<br>
    1. Deteksi anomali (ML)<br>
    2. Pola clustering<br>
    3. Standar pupuk (jika aktif)<br>
    4. Intensitas penggunaan per hektar
    </div>
    """, unsafe_allow_html=True)
    
    # Summary by priority
    st.subheader("📊 Ringkasan Prioritas")
    
    col1, col2, col3 = st.columns(3)
    
    tinggi = (df['Prioritas'] == 'Tinggi').sum()
    sedang = (df['Prioritas'] == 'Sedang').sum()
    rendah = (df['Prioritas'] == 'Rendah').sum()
    
    with col1:
        st.markdown(f"""
        <div class="danger-box">
        <h3>🚨 Prioritas Tinggi</h3>
        <h2>{tinggi:,} petani</h2>
        <p>{tinggi/len(df)*100:.1f}% dari total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="warning-box">
        <h3>⚠️ Prioritas Sedang</h3>
        <h2>{sedang:,} petani</h2>
        <p>{sedang/len(df)*100:.1f}% dari total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="success-box">
        <h3>✅ Prioritas Rendah</h3>
        <h2>{rendah:,} petani</h2>
        <p>{rendah/len(df)*100:.1f}% dari total</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🔍 Detail Rekomendasi")
    
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
    
    st.info(f"📊 Menampilkan {len(df_filtered):,} petani")
    
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
        label="⬇️ Download Rekomendasi (CSV)",
        data=csv,
        file_name=f"rdkk_rekomendasi_{priority_filter.lower()}.csv",
        mime="text/csv",
    )
    
    # Action plan summary
    if 'Action_Plan' in df_filtered.columns:
        st.markdown("---")
        st.subheader("📋 Action Plan Prioritas Tinggi")
        
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
elif page == "📚 Tentang Model":
    st.markdown('<div class="section-header">📚 Tentang Model Machine Learning</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>ℹ️ Penjelasan untuk User Awam:</strong><br>
    Halaman ini menjelaskan bagaimana sistem menggunakan kecerdasan buatan (AI) untuk menganalisis data pupuk subsidi.
    Tidak perlu keahlian teknis - kami jelaskan dengan bahasa sederhana!
    </div>
    """, unsafe_allow_html=True)
    
    # Section 1: Ringkasan Model
    st.markdown("---")
    st.markdown("### 🤖 1. Ringkasan Model AI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <strong>🔍 Isolation Forest (Deteksi Anomali)</strong>
        
        Model ini bekerja seperti "pendeteksi pola aneh":
        - Menganalisis semua data penggunaan pupuk
        - Mencari pola yang tidak biasa atau ekstrem
        - Menandai petani dengan penggunaan yang sangat berbeda dari mayoritas
        - Bekerja otomatis tanpa perlu aturan manual
        
        <strong>Kenapa penting?</strong>  
        Membantu menemukan kasus overuse atau underuse yang signifikan secara objektif berdasarkan data aktual.
        """)
        
        if df is not None and 'Anomaly_Label' in df.columns:
            anomaly_counts = df['Anomaly_Label'].value_counts()
            normal_count = anomaly_counts.get('Normal', 0)
            anomaly_count = len(df) - normal_count
            
            st.metric("Total Data Dianalisis", f"{len(df):,} petani")
            st.metric("Deteksi Normal", f"{normal_count:,}", f"{normal_count/len(df)*100:.1f}%")
            st.metric("Deteksi Anomali", f"{anomaly_count:,}", f"{anomaly_count/len(df)*100:.1f}%")
    
    with col2:
        st.markdown("""
        <strong>🎯 KMeans Clustering (Pengelompokan)</strong>
        
        Model ini seperti "pengelompokan otomatis":
        - Mengelompokkan petani dengan pola serupa
        - Berdasarkan luas lahan, intensitas pupuk, dan pola penggunaan
        - Membantu memahami segmen petani yang berbeda
        - Berguna untuk strategi distribusi yang lebih tepat sasaran
        
        <strong>Kenapa penting?</strong>  
        Membantu pemerintah memahami berbagai tipe petani dan kebutuhan mereka yang berbeda-beda.
        """)
        
        if df is not None and 'Cluster_ID' in df.columns:
            n_clusters = df['Cluster_ID'].nunique()
            st.metric("Jumlah Kelompok", f"{n_clusters} cluster")
            
            if models and 'clustering' in models:
                feature_count = len(models['clustering'].get('feature_cols', []))
                st.metric("Fitur yang Dianalisis", f"{feature_count} variabel")
    
    # Section 2: Pipeline Diagram
    st.markdown("---")
    st.markdown("### 🔄 2. Alur Kerja Sistem (Pipeline)")
    
    st.markdown("""
    Berikut adalah tahapan pemrosesan data dari awal hingga menghasilkan rekomendasi:
    """)
    
    pipeline_cols = st.columns(7)
    
    pipeline_steps = [
        ("📥", "1. Load Data", "Membaca data CSV petani", "#e3f2fd"),
        ("🧹", "2. Preprocessing", "Membersihkan & standarisasi data", "#f3e5f5"),
        ("⚙️", "3. Feature Engineering", "Menghitung pupuk per ha, total, dll", "#e8f5e9"),
        ("🔍", "4. Anomaly Detection", "Deteksi pola tidak normal (ML)", "#fff3e0"),
        ("🎯", "5. Clustering", "Pengelompokan pola serupa", "#fce4ec"),
        ("💡", "6. Generate Recommendations", "Buat rekomendasi & prioritas", "#e0f2f1"),
        ("💾", "7. Export Results", "Simpan hasil analisis", "#f1f8e9")
    ]
    
    for i, (icon, title, desc, color) in enumerate(pipeline_steps):
        with pipeline_cols[i]:
            st.markdown(f"""
            <div style="background: {color}; padding: 1rem; border-radius: 0.5rem; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 2rem;">{icon}</div>
            <strong>{title}</strong>
            <p style="font-size: 0.8rem; margin-top: 0.5rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Section 3: Cara Model Mengambil Keputusan
    st.markdown("---")
    st.markdown("### 🧠 3. Cara Model Mengambil Keputusan")
    
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["🔍 Anomaly Detection", "🎯 Clustering", "💡 Rekomendasi"])
        
        with tab1:
            st.markdown("""
            <strong>Bagaimana Isolation Forest Mendeteksi Anomali?</strong>
            
            Model ini menggunakan konsep "isolasi":
            1. Data normal biasanya berkelompok dan sulit dipisahkan
            2. Data anomali biasanya terisolasi dan mudah dipisahkan
            3. Model menghitung "skor anomali" untuk setiap petani
            4. Skor tinggi = kemungkinan besar anomali (pola tidak biasa)
            """)
            
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
                    st.markdown("<strong>Penjelasan Kategori:</strong>")
                    for label in anomaly_dist.index:
                        count = anomaly_dist[label]
                        pct = count / len(df) * 100
                        
                        if label == 'Normal':
                            st.markdown(f"✅ <strong>{label}</strong>: {count:,} petani ({pct:.1f}%)")
                            st.markdown("Penggunaan pupuk dalam batas wajar")
                        elif label == 'Ringan':
                            st.markdown(f"⚠️ <strong>{label}</strong>: {count:,} petani ({pct:.1f}%)")
                            st.markdown("Sedikit menyimpang, perlu perhatian")
                        else:
                            st.markdown(f"🚨 <strong>{label}</strong>: {count:,} petani ({pct:.1f}%)")
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
                    
                    with st.expander(f"Cluster {cluster_id} ({len(cluster_data)} petani)", expanded=False):
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
    st.markdown("### 📊 4. Fitur-Fitur Penting yang Dianalisis")
    
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
    st.markdown("### 📈 5. Statistik & Performa Model")
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("##### 📊 Data")
            st.metric("Total Sampel", f"{len(df):,}")
            
            if 'Komoditas' in df.columns:
                st.metric("Jenis Komoditas", df['Komoditas'].nunique())
            
            if 'Desa' in df.columns:
                st.metric("Jumlah Desa", df['Desa'].nunique())
        
        with col2:
            st.markdown("##### 🎯 Clustering")
            
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
            st.markdown("##### 🔍 Anomaly")
            
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
            st.markdown("##### 📏 Deviasi Pupuk")
            
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
    st.markdown("### 💡 6. Kesimpulan & Insight Otomatis")
    
    if df is not None:
        st.markdown("<strong>Berdasarkan analisis data, sistem menghasilkan insight berikut:</strong>", unsafe_allow_html=True)
        
        insights = []
        
        # Insight 1: Anomaly rate
        if 'Anomaly_Label' in df.columns:
            anomaly_count = (df['Anomaly_Label'] != 'Normal').sum()
            anomaly_pct = anomaly_count / len(df) * 100
            
            if anomaly_pct > 20:
                insights.append(f"🚨 <strong>Tingkat anomali cukup tinggi ({anomaly_pct:.1f}%)</strong> - Perlu investigasi mendalam pada {anomaly_count:,} petani yang terdeteksi memiliki pola penggunaan tidak normal.")
            elif anomaly_pct > 10:
                insights.append(f"⚠️ <strong>Tingkat anomali moderat ({anomaly_pct:.1f}%)</strong> - Ada {anomaly_count:,} petani yang perlu perhatian khusus.")
            else:
                insights.append(f"✅ <strong>Tingkat anomali rendah ({anomaly_pct:.1f}%)</strong> - Sebagian besar petani ({len(df)-anomaly_count:,}) memiliki pola penggunaan normal.")
        
        # Insight 2: Clustering distribution
        if 'Cluster_ID' in df.columns:
            n_clusters = df['Cluster_ID'].nunique()
            insights.append(f"🎯 <strong>Teridentifikasi {n_clusters} kelompok petani berbeda</strong> - Menunjukkan keberagaman pola penggunaan pupuk yang perlu pendekatan berbeda.")
            
            # Most common cluster
            most_common_cluster = df['Cluster_ID'].mode()[0]
            most_common_count = (df['Cluster_ID'] == most_common_cluster).sum()
            insights.append(f"📊 <strong>Cluster {most_common_cluster} adalah yang terbesar</strong> dengan {most_common_count:,} petani ({most_common_count/len(df)*100:.1f}%), menunjukkan pola dominan.")
        
        # Insight 3: Priority distribution
        if 'Prioritas' in df.columns:
            high_priority = (df['Prioritas'] == 'Tinggi').sum()
            high_pct = high_priority / len(df) * 100
            
            if high_pct > 15:
                insights.append(f"🚨 <strong>{high_priority:,} petani ({high_pct:.1f}%) memerlukan tindakan segera</strong> - Prioritas tinggi untuk intervensi.")
            elif high_pct > 5:
                insights.append(f"⚠️ <strong>{high_priority:,} petani ({high_pct:.1f}%) perlu perhatian</strong> - Monitoring lebih intensif diperlukan.")
            else:
                insights.append(f"✅ <strong>Mayoritas petani dalam kondisi baik</strong> - Hanya {high_priority:,} ({high_pct:.1f}%) yang perlu tindakan segera.")
        
        # Insight 4: Commodity-specific
        if 'Komoditas' in df.columns and 'Total_per_ha' in df.columns:
            komoditas_intensity = df.groupby('Komoditas')['Total_per_ha'].mean().sort_values(ascending=False)
            highest_komoditas = komoditas_intensity.index[0]
            highest_value = komoditas_intensity.iloc[0]
            lowest_komoditas = komoditas_intensity.index[-1]
            lowest_value = komoditas_intensity.iloc[-1]
            
            insights.append(f"🌾 <strong>{highest_komoditas} memiliki intensitas pupuk tertinggi</strong> ({highest_value:.1f} kg/ha), sedangkan <strong>{lowest_komoditas} terendah</strong> ({lowest_value:.1f} kg/ha).")
        
        # Insight 5: Standards compliance (if enabled)
        if standards_enabled and 'Final_Status' in df.columns:
            overuse_count = (df['Final_Status'] == 'Overuse').sum()
            underuse_count = (df['Final_Status'] == 'Underuse').sum()
            normal_count = (df['Final_Status'] == 'Normal').sum()
            overuse_pct = overuse_count / len(df) * 100
            underuse_pct = underuse_count / len(df) * 100
            normal_pct = normal_count / len(df) * 100
            
            if overuse_pct > underuse_pct:
                insights.append(f"📈 <strong>Overuse lebih dominan ({overuse_pct:.1f}%) dibanding underuse ({underuse_pct:.1f}%)</strong> - Fokus pada edukasi pengurangan dosis dan efisiensi penggunaan.")
            elif underuse_pct > overuse_pct:
                insights.append(f"📉 <strong>Underuse lebih dominan ({underuse_pct:.1f}%) dibanding overuse ({overuse_pct:.1f}%)</strong> - Fokus pada peningkatan akses dan edukasi manfaat pupuk optimal.")
            else:
                insights.append(f"⚖️ <strong>Distribusi relatif seimbang</strong>: Normal {normal_pct:.1f}%, Overuse {overuse_pct:.1f}%, Underuse {underuse_pct:.1f}%")
        
        # Insight 6: Data quality
        if 'Total_Pupuk' in df.columns:
            zero_pupuk = (df['Total_Pupuk'] == 0).sum()
            if zero_pupuk > 0:
                insights.append(f"⚠️ <strong>{zero_pupuk:,} petani memiliki total pupuk = 0</strong> - Perlu verifikasi data atau memang tidak menerima pupuk.")
        
        # Display insights
        for i, insight in enumerate(insights, 1):
            st.markdown(f"{i}. {insight}", unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📋 Rekomendasi Umum Berdasarkan Analisis")
        
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
            recommendations_general.append(f"5. <strong>Strategi distribusi berbeda per cluster</strong> - {df['Cluster_ID'].nunique()} segmen petani memerlukan pendekatan berbeda")
        
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
# PAGE 7: KELOLA STANDAR
# ==========================================
elif page == "⚙️ Kelola Standar":
    st.markdown('<div class="section-header">⚙️ Kelola Standar Pupuk</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>ℹ️ Pengelolaan Standar:</strong><br>
    Anda dapat menambah, edit, atau hapus standar pupuk per komoditas.
    Standar ini digunakan untuk menentukan status Underuse/Overuse.
    </div>
    """, unsafe_allow_html=True)
    
    # Toggle status
    st.markdown(f"<strong>Status Saat Ini:</strong> {'🟢 Aktif' if standards_enabled else '🔴 Non-aktif'}", unsafe_allow_html=True)
    
    # Tampilkan standar yang ada
    st.subheader("📋 Standar Pupuk Saat Ini")
    
    all_standards = standards_manager.get_all_standards()
    
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
    else:
        st.warning("⚠️ Belum ada standar pupuk yang terdaftar")
    
    # Form untuk tambah/edit standar
    st.markdown("---")
    st.subheader("➕ Tambah/Edit Standar")
    
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
        st.subheader("🗑️ Hapus Standar")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            komoditas_to_delete = st.selectbox(
                "Pilih komoditas untuk dihapus:",
                list(all_standards.keys())
            )
        
        with col2:
            st.write("")  # Spacing
            if st.button("🗑️ Hapus", type="secondary", use_container_width=True):
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
    <p><strong>RDKK System v4.0</strong> | Powered by Machine Learning & Data Science</p>
    <p>© 2025 - Sistem Analisis Pupuk Subsidi</p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">
        🌾 Membantu petani mengoptimalkan penggunaan pupuk untuk hasil panen yang lebih baik
    </p>
</div>
""", unsafe_allow_html=True)
