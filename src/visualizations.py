"""
Visualizations Module
Fungsi-fungsi untuk visualisasi data dengan Plotly
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_distribution_pupuk(df: pd.DataFrame, pupuk_type: str = 'Total_per_ha'):
    """
    Plot distribusi penggunaan pupuk
    
    Args:
        df: DataFrame dengan data
        pupuk_type: 'Total_per_ha', 'Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha'
    """
    fig = px.histogram(
        df, 
        x=pupuk_type,
        nbins=30,
        title=f'Distribusi {pupuk_type.replace("_", " ").title()}',
        labels={pupuk_type: 'Jumlah (kg/ha)', 'count': 'Frekuensi'},
        color_discrete_sequence=['#3b82f6']
    )
    
    fig.update_layout(
        showlegend=False,
        height=400
    )
    
    return fig

def plot_anomaly_scatter(df: pd.DataFrame):
    """
    Scatter plot luas lahan vs total pupuk dengan warna anomali
    """
    color_map = {
        'normal': '#22c55e',
        'underuse': '#3b82f6',
        'overuse': '#ef4444',
        'outlier_pola': '#f59e0b'
    }
    
    fig = px.scatter(
        df,
        x='Luas_Tanah_ha',
        y='Total_per_ha',
        color='Final_Anomaly_Category',
        color_discrete_map=color_map,
        title='Hubungan Luas Lahan vs Penggunaan Pupuk',
        labels={
            'Luas_Tanah_ha': 'Luas Lahan (ha)',
            'Total_per_ha': 'Total Pupuk per ha (kg)',
            'Final_Anomaly_Category': 'Kategori'
        },
        hover_data=['Komoditas', 'Desa']
    )
    
    fig.update_layout(height=500)
    
    return fig

def plot_cluster_2d(df: pd.DataFrame):
    """
    Visualisasi cluster dalam 2D (PCA atau pilih 2 fitur utama)
    """
    fig = px.scatter(
        df,
        x='Luas_Tanah_ha',
        y='Total_per_ha',
        color='Cluster_Label',
        title='Visualisasi Cluster',
        labels={
            'Luas_Tanah_ha': 'Luas Lahan (ha)',
            'Total_per_ha': 'Total Pupuk per ha (kg)',
            'Cluster_Label': 'Cluster'
        },
        hover_data=['Komoditas', 'Final_Anomaly_Category']
    )
    
    fig.update_layout(height=500)
    
    return fig

def plot_komoditas_distribution(df: pd.DataFrame):
    """
    Bar chart distribusi per komoditas
    """
    komoditas_counts = df['Komoditas'].value_counts().reset_index()
    komoditas_counts.columns = ['Komoditas', 'Jumlah']
    
    fig = px.bar(
        komoditas_counts,
        x='Komoditas',
        y='Jumlah',
        title='Distribusi Petani per Komoditas',
        color='Jumlah',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def plot_boxplot_by_commodity(df: pd.DataFrame, metric: str = 'Total_per_ha'):
    """
    Boxplot distribusi pupuk per komoditas
    """
    fig = px.box(
        df,
        x='Komoditas',
        y=metric,
        title=f'Distribusi {metric.replace("_", " ").title()} per Komoditas',
        labels={
            'Komoditas': 'Komoditas',
            metric: 'Jumlah (kg/ha)'
        },
        color='Komoditas'
    )
    
    fig.update_layout(height=500, showlegend=False)
    
    return fig

def plot_heatmap_desa(df: pd.DataFrame):
    """
    Heatmap rata-rata penggunaan pupuk per desa
    """
    # Aggregate by desa
    desa_stats = df.groupby('Desa').agg({
        'Urea_per_ha': 'mean',
        'NPK_per_ha': 'mean',
        'Organik_per_ha': 'mean'
    }).round(1)
    
    fig = go.Figure(data=go.Heatmap(
        z=desa_stats.values.T,
        x=desa_stats.index,
        y=['Urea', 'NPK', 'Organik'],
        colorscale='Blues',
        text=desa_stats.values.T,
        texttemplate='%{text:.0f}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Rata-rata Penggunaan Pupuk per Desa (kg/ha)',
        xaxis_title='Desa',
        yaxis_title='Jenis Pupuk',
        height=400
    )
    
    return fig

def plot_quota_vs_usage(df: pd.DataFrame):
    """
    Perbandingan jatah vs penggunaan aktual
    """
    # Filter hanya yang punya quota
    df_with_quota = df[df['Total_Quota'] > 0].copy()
    
    if len(df_with_quota) == 0:
        # Return empty plot
        fig = go.Figure()
        fig.add_annotation(
            text="Tidak ada data dengan standar pupuk",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Aggregate by komoditas
    comparison = df_with_quota.groupby('Komoditas').agg({
        'Total_Quota': 'mean',
        'Total_Pupuk': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Jatah',
        x=comparison['Komoditas'],
        y=comparison['Total_Quota'],
        marker_color='#94a3b8'
    ))
    
    fig.add_trace(go.Bar(
        name='Penggunaan Aktual',
        x=comparison['Komoditas'],
        y=comparison['Total_Pupuk'],
        marker_color='#3b82f6'
    ))
    
    fig.update_layout(
        title='Perbandingan Jatah vs Penggunaan Aktual per Komoditas',
        xaxis_title='Komoditas',
        yaxis_title='Total Pupuk (kg)',
        barmode='group',
        height=400
    )
    
    return fig

def plot_radar_cluster(df: pd.DataFrame, cluster_id: int):
    """
    Radar chart untuk karakteristik cluster tertentu
    """
    cluster_data = df[df['Cluster_ID'] == cluster_id]
    
    metrics = ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha', 'Luas_Tanah_ha']
    available_metrics = [m for m in metrics if m in cluster_data.columns]
    
    values = [cluster_data[m].mean() for m in available_metrics]
    
    # Normalize to 0-1 scale for better visualization
    max_values = [df[m].max() for m in available_metrics]
    normalized_values = [v/m if m > 0 else 0 for v, m in zip(values, max_values)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=[m.replace('_', ' ').title() for m in available_metrics],
        fill='toself',
        name=f'Cluster {cluster_id}'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=f'Karakteristik Cluster {cluster_id}',
        height=400
    )
    
    return fig
