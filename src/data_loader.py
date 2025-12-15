"""
Data Loader Module
Memuat dan validasi data RDKK
"""

import pandas as pd
import numpy as np
from typing import Optional

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data dari CSV atau Excel
    
    Args:
        file_path: Path ke file data
    
    Returns:
        DataFrame dengan data RDKK
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Format file tidak didukung. Gunakan CSV atau Excel.")
    
    return df

def validate_columns(df: pd.DataFrame) -> tuple[bool, list]:
    """
    Validasi kolom yang diperlukan ada di DataFrame
    
    Returns:
        (is_valid, missing_columns)
    """
    required_columns = [
        'ID_Petani', 'Desa', 'Kelompok_Tani', 'Komoditas', 'Luas_Tanah_m2',
        'Urea_MT1', 'NPK_MT1', 'Organik_MT1',
        'Urea_MT2', 'NPK_MT2', 'Organik_MT2',
        'Urea_MT3', 'NPK_MT3', 'Organik_MT3'
    ]
    
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Persiapan data: copy dan convert tipe data
    """
    df = df.copy()
    
    # Pastikan kolom numerik adalah float
    numeric_cols = [
        'Luas_Tanah_m2',
        'Urea_MT1', 'NPK_MT1', 'Organik_MT1',
        'Urea_MT2', 'NPK_MT2', 'Organik_MT2',
        'Urea_MT3', 'NPK_MT3', 'Organik_MT3'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df
