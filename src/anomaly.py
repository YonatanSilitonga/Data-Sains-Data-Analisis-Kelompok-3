"""
Anomaly Detection Module
UPDATED: Deteksi anomali HANYA berdasarkan pola distribusi total pupuk per ha
TIDAK lagi menggunakan MT, proporsi MT, atau quota berbasis standar untuk anomaly
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Optional, Tuple

def detect_anomaly_isolation_forest(
    df: pd.DataFrame, 
    contamination: float = 0.1,
    features: Optional[list] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Deteksi anomali menggunakan Isolation Forest
    UPDATED: Fokus HANYA pada total pupuk per jenis per hektar
    
    Args:
        df: DataFrame dengan kolom Urea_per_ha, NPK_per_ha, Organik_per_ha
        contamination: Expected proportion of anomalies (0.05-0.15)
        features: List fitur untuk deteksi (default: total per ha per jenis)
    
    Returns:
        (DataFrame dengan hasil anomaly, model_info dict)
    """
    df = df.copy()
    
    if features is None:
        features = ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha']
    
    # Validasi fitur tersedia
    available_features = [f for f in features if f in df.columns]
    if not available_features:
        raise ValueError(f"Tidak ada fitur yang tersedia dari: {features}")
    
    X = df[available_features].fillna(0).values
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        max_samples='auto'
    )
    
    predictions = iso_forest.fit_predict(X)
    scores = iso_forest.score_samples(X)
    
    df['Anomaly_Prediction'] = predictions
    df['Anomaly_Score'] = -scores  # Negatif agar semakin tinggi = semakin anomali
    df['Anomaly_Label'] = df['Anomaly_Prediction'].apply(
        lambda x: 'Anomali' if x == -1 else 'Normal'
    )
    
    model_info = {
        'model': iso_forest,
        'features': available_features,
        'contamination': contamination,
        'n_anomalies': (predictions == -1).sum(),
        'n_normal': (predictions == 1).sum()
    }
    
    return df, model_info

def detect_anomaly_per_commodity(df: pd.DataFrame, contamination: float = 0.1) -> pd.DataFrame:
    """
    Deteksi anomali per komoditas (lebih spesifik)
    UPDATED: Menggunakan median komoditas sebagai baseline
    
    Petani dianggap anomali jika berbeda signifikan dari mayoritas petani komoditas yang sama
    """
    df = df.copy()
    df['Anomaly_Label'] = 'Unknown'
    df['Anomaly_Score'] = 0.0
    df['Anomaly_Prediction'] = 1
    
    features = ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha']
    
    for komoditas in df['Komoditas'].unique():
        mask = df['Komoditas'] == komoditas
        komoditas_df = df[mask].copy()
        
        if len(komoditas_df) < 10:
            df.loc[mask, 'Anomaly_Label'] = 'Insufficient Data'
            continue
        
        # Deteksi anomali dalam komoditas
        available_features = [f for f in features if f in komoditas_df.columns]
        X = komoditas_df[available_features].fillna(0).values
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        predictions = iso_forest.fit_predict(X)
        scores = iso_forest.score_samples(X)
        
        df.loc[mask, 'Anomaly_Prediction'] = predictions
        df.loc[mask, 'Anomaly_Score'] = -scores
        df.loc[mask, 'Anomaly_Label'] = [
            'Anomali' if p == -1 else 'Normal' for p in predictions
        ]
    
    return df

def compare_with_median_distribution(df: pd.DataFrame, tolerance: float = 0.3) -> pd.DataFrame:
    """
    Bandingkan dengan distribusi median per komoditas
    UPDATED: Menggunakan median ± 30% sebagai rentang normal
    
    Args:
        df: DataFrame dengan total per ha
        tolerance: Toleransi dari median (0.3 = 30%)
    
    Returns:
        DataFrame dengan label Median_Status per pupuk
    """
    df = df.copy()
    
    pupuk_cols = ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha']
    
    for komoditas in df['Komoditas'].unique():
        mask = df['Komoditas'] == komoditas
        komoditas_df = df[mask]
        
        for pupuk_col in pupuk_cols:
            if pupuk_col not in komoditas_df.columns:
                continue
            
            median_val = komoditas_df[pupuk_col].median()
            
            if median_val == 0:
                df.loc[mask, f'Median_Label_{pupuk_col.split("_")[0]}'] = 'Unknown'
                continue
            
            lower_bound = median_val * (1 - tolerance)
            upper_bound = median_val * (1 + tolerance)
            
            def categorize(val):
                if pd.isna(val):
                    return 'Unknown'
                elif val < lower_bound:
                    return 'Underuse'
                elif val > upper_bound:
                    return 'Overuse'
                else:
                    return 'Normal'
            
            label_col = f'Median_Label_{pupuk_col.split("_")[0]}'
            df.loc[mask, label_col] = komoditas_df[pupuk_col].apply(categorize)
    
    def get_overall_status(row):
        labels = []
        for pupuk in ['Urea', 'NPK', 'Organik']:
            col = f'Median_Label_{pupuk}'
            if col in row.index:
                labels.append(row[col])
        
        if 'Unknown' in labels or not labels:
            return 'Unknown'
        
        # Jika ada overuse, prioritaskan
        if 'Overuse' in labels:
            return 'Overuse'
        elif 'Underuse' in labels:
            return 'Underuse'
        else:
            return 'Normal'
    
    df['Median_Status'] = df.apply(get_overall_status, axis=1)
    
    return df

def run_anomaly_detection_pipeline(
    df: pd.DataFrame, 
    contamination: float = 0.1,
    use_per_commodity: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """
    Pipeline lengkap anomaly detection dengan konsep baru
    
    Args:
        df: DataFrame dengan fitur total per ha
        contamination: Expected anomaly proportion
        use_per_commodity: Deteksi per komoditas atau global
    
    Returns:
        (DataFrame dengan hasil, model_info)
    """
    
    if use_per_commodity:
        # Deteksi per komoditas (lebih akurat)
        df = detect_anomaly_per_commodity(df, contamination)
        model_info = {
            'method': 'per_commodity',
            'contamination': contamination
        }
    else:
        # Deteksi global
        df, model_info = detect_anomaly_isolation_forest(df, contamination)
        model_info['method'] = 'global'
    
    # Tambahkan perbandingan dengan median
    df = compare_with_median_distribution(df)
    
    model_info['n_total'] = len(df)
    model_info['n_anomali'] = (df['Anomaly_Label'] == 'Anomali').sum()
    model_info['n_normal'] = (df['Anomaly_Label'] == 'Normal').sum()
    model_info['pct_anomali'] = model_info['n_anomali'] / model_info['n_total'] * 100
    
    return df, model_info

def print_anomaly_summary(model_info: dict):
    """Print ringkasan hasil deteksi anomali"""
    print("\n" + "="*60)
    print("RINGKASAN DETEKSI ANOMALI")
    print("="*60)
    print(f"Metode: {model_info.get('method', 'unknown')}")
    print(f"Total data: {model_info.get('n_total', 0):,}")
    print(f"  - Anomali: {model_info.get('n_anomali', 0):,} ({model_info.get('pct_anomali', 0):.1f}%)")
    print(f"  - Normal: {model_info.get('n_normal', 0):,}")
    print("="*60 + "\n")
