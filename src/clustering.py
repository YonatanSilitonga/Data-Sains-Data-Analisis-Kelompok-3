"""
Clustering Module
Clustering pola penggunaan pupuk menggunakan KMeans atau Gaussian Mixture
UPDATED: Clustering boleh gunakan MT sebagai KARAKTER, bukan penilaian
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Tuple, Optional

def find_optimal_k(X: np.ndarray, max_k: int = 10, method: str = 'silhouette', random_state: int = 42) -> int:
    """
    Tentukan jumlah cluster optimal menggunakan elbow/silhouette
    
    Args:
        X: Feature matrix
        max_k: Max jumlah cluster untuk test
        method: 'silhouette' atau 'elbow'
        random_state: seed untuk hasil deterministik
    
    Returns:
        Optimal k
    """
    if len(X) < max_k:
        max_k = len(X) - 1
    
    scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
    
    # Pilih k dengan silhouette score tertinggi
    optimal_k = K_range[np.argmax(scores)]
    
    return optimal_k

def prepare_clustering_features(df: pd.DataFrame) -> Tuple[np.ndarray, list]:
    """
    Persiapkan features untuk clustering
    UPDATED: Gunakan total per ha sebagai fitur utama
    MT bisa digunakan sebagai karakter tambahan (opsional)
    """
    feature_cols = [
        'Luas_Tanah_ha',
        'Urea_per_ha',  # Total Urea per ha
        'NPK_per_ha',   # Total NPK per ha
        'Organik_per_ha',  # Total Organik per ha
    ]
    
    # Filter kolom yang ada
    available_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[available_cols].fillna(0).values
    
    return X, available_cols

def train_kmeans_clustering(df: pd.DataFrame, n_clusters: Optional[int] = None, random_state: int = 42) -> Tuple[pd.DataFrame, dict]:
    """
    Train KMeans clustering dengan auto k-selection
    
    Args:
        df: DataFrame dengan features
        n_clusters: jumlah cluster (jika None, akan dicari otomatis)
        random_state: seed untuk hasil deterministik
    
    Returns:
        (df_with_clusters, model_info)
    """
    df = df.copy()
    
    # Prepare features
    X, feature_cols = prepare_clustering_features(df)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal k if not specified
    if n_clusters is None:
        n_clusters = find_optimal_k(X_scaled, random_state=random_state)
    
    # Train KMeans dengan random_state untuk hasil konsisten
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(X_scaled)
    
    # Evaluate
    silhouette = silhouette_score(X_scaled, df['Cluster_ID'])
    davies_bouldin = davies_bouldin_score(X_scaled, df['Cluster_ID'])
    
    # Generate cluster labels
    df['Cluster_Label'] = df['Cluster_ID'].apply(lambda x: f'Cluster {x}')
    
    model_info = {
        'model': kmeans,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'n_clusters': n_clusters,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin
    }
    
    return df, model_info

def generate_cluster_characteristics(df: pd.DataFrame) -> dict:
    """
    Generate penjelasan karakteristik setiap cluster secara otomatis
    UPDATED: Penjelasan TIDAK menyebut MT dalam deskripsi
    
    Returns:
        dict dengan cluster_id sebagai key dan deskripsi sebagai value
    """
    characteristics = {}
    
    for cluster_id in sorted(df['Cluster_ID'].unique()):
        cluster_data = df[df['Cluster_ID'] == cluster_id]
        
        # Hitung statistik
        avg_luas = cluster_data['Luas_Tanah_ha'].mean()
        avg_urea = cluster_data['Urea_per_ha'].mean() if 'Urea_per_ha' in cluster_data.columns else 0
        avg_npk = cluster_data['NPK_per_ha'].mean() if 'NPK_per_ha' in cluster_data.columns else 0
        avg_organik = cluster_data['Organik_per_ha'].mean() if 'Organik_per_ha' in cluster_data.columns else 0
        
        if 'Anomaly_Label' in cluster_data.columns:
            anomali_pct = (cluster_data['Anomaly_Label'] == 'Anomali').mean() * 100
        else:
            anomali_pct = 0
        
        # Generate deskripsi (TIDAK menyebut MT)
        description = f"Cluster {cluster_id}: "
        
        # Karakteristik luas lahan
        if avg_luas < 0.5:
            description += "Petani lahan kecil (<0.5 ha), "
        elif avg_luas < 1.5:
            description += "Petani lahan sedang (0.5-1.5 ha), "
        else:
            description += "Petani lahan besar (>1.5 ha), "
        
        # Karakteristik penggunaan pupuk per ha
        total_avg = avg_urea + avg_npk + avg_organik
        if total_avg < 300:
            description += "intensitas pupuk rendah"
        elif total_avg < 700:
            description += "intensitas pupuk sedang"
        else:
            description += "intensitas pupuk tinggi"
        
        # Pola anomali
        if anomali_pct > 30:
            description += f", banyak anomali ({anomali_pct:.0f}%)"
        elif anomali_pct < 10:
            description += ", pola penggunaan stabil"
        
        characteristics[str(cluster_id)] = description
    
    return characteristics

def run_clustering_pipeline(df: pd.DataFrame, n_clusters: Optional[int] = None, random_state: int = 42) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Pipeline lengkap clustering
    
    Args:
        df: DataFrame dengan features
        n_clusters: jumlah cluster (jika None, akan dicari otomatis)
        random_state: seed untuk hasil deterministik
    
    Returns:
        (df_with_clusters, model_info, characteristics)
    """
    df_clustered, model_info = train_kmeans_clustering(df, n_clusters, random_state)
    characteristics = generate_cluster_characteristics(df_clustered)
    
    return df_clustered, model_info, characteristics
