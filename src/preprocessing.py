"""
Preprocessing Module
Membersihkan dan memproses data sebelum analisis
"""

import pandas as pd
import numpy as np
from typing import Optional

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan data: handle missing values, outliers, dll
    """
    df = df.copy()
    
    # Kolom pupuk
    fertilizer_cols = [
        'Urea_MT1', 'NPK_MT1', 'Organik_MT1',
        'Urea_MT2', 'NPK_MT2', 'Organik_MT2',
        'Urea_MT3', 'NPK_MT3', 'Organik_MT3'
    ]
    
    # Fill missing values dengan 0 untuk kolom pupuk
    for col in fertilizer_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Fill missing Luas_Tanah_m2 dengan median
    if 'Luas_Tanah_m2' in df.columns:
        df['Luas_Tanah_m2'] = df['Luas_Tanah_m2'].fillna(df['Luas_Tanah_m2'].median())
    
    # Remove rows dengan Luas_Tanah_m2 <= 0
    df = df[df['Luas_Tanah_m2'] > 0]
    
    # Remove extreme outliers (values > 99.9 percentile)
    for col in fertilizer_cols:
        if col in df.columns:
            threshold = df[col].quantile(0.999)
            df = df[df[col] <= threshold]
    
    return df

def calculate_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung total pupuk per jenis dari semua MT
    """
    df = df.copy()
    
    # Total per jenis pupuk
    df['Total_Urea'] = df['Urea_MT1'] + df['Urea_MT2'] + df['Urea_MT3']
    df['Total_NPK'] = df['NPK_MT1'] + df['NPK_MT2'] + df['NPK_MT3']
    df['Total_Organik'] = df['Organik_MT1'] + df['Organik_MT2'] + df['Organik_MT3']
    
    # Total semua pupuk
    df['Total_Pupuk'] = df['Total_Urea'] + df['Total_NPK'] + df['Total_Organik']
    
    return df

def calculate_per_ha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konversi luas tanah ke hektar
    NOTE: Perhitungan pupuk per ha DIPINDAHKAN ke feature_engineering
    """
    df = df.copy()
    
    df['Luas_Tanah_ha'] = df['Luas_Tanah_m2'] / 10000
    
    return df

def apply_standards_labeling(df: pd.DataFrame, standards_manager) -> pd.DataFrame:
    """
    Apply standar pupuk dan labelkan status over/under/normal use
    
    Args:
        df: DataFrame dengan Total_Urea, Total_NPK, Total_Organik
        standards_manager: Instance dari StandardsManager
    
    Returns:
        DataFrame dengan kolom status tambahan
    """
    df = df.copy()
    
    # Initialize status columns
    df['Status_Urea'] = 'Unknown'
    df['Status_NPK'] = 'Unknown'
    df['Status_Organik'] = 'Unknown'
    df['Jatah_Urea_Min'] = 0.0
    df['Jatah_Urea_Max'] = 0.0
    df['Jatah_NPK_Min'] = 0.0
    df['Jatah_NPK_Max'] = 0.0
    df['Jatah_Organik_Min'] = 0.0
    df['Jatah_Organik_Max'] = 0.0
    
    # Apply standards per row
    for idx, row in df.iterrows():
        komoditas = row['Komoditas']
        luas_m2 = row['Luas_Tanah_m2']
        total_urea = row['Total_Urea']
        total_npk = row['Total_NPK']
        total_organik = row['Total_Organik']
        
        result = standards_manager.calculate_status(
            komoditas, luas_m2, total_urea, total_npk, total_organik
        )
        
        df.at[idx, 'Status_Urea'] = result['status_urea']
        df.at[idx, 'Status_NPK'] = result['status_npk']
        df.at[idx, 'Status_Organik'] = result['status_organik']
        df.at[idx, 'Jatah_Urea_Min'] = result['jatah_urea_min']
        df.at[idx, 'Jatah_Urea_Max'] = result['jatah_urea_max']
        df.at[idx, 'Jatah_NPK_Min'] = result['jatah_npk_min']
        df.at[idx, 'Jatah_NPK_Max'] = result['jatah_npk_max']
        df.at[idx, 'Jatah_Organik_Min'] = result['jatah_organik_min']
        df.at[idx, 'Jatah_Organik_Max'] = result['jatah_organik_max']
    
    # Calculate overall status (kombinasi dari ketiga pupuk)
    def get_overall_status(row):
        statuses = [row['Status_Urea'], row['Status_NPK'], row['Status_Organik']]
        
        # Jika ada Unknown, return Unknown
        if 'Unknown' in statuses:
            return 'Unknown'
        
        # Jika semua normal, return Normal
        if all(s == 'Normal' for s in statuses):
            return 'Normal'
        
        # Jika ada overuse, prioritaskan overuse
        if 'Overuse' in statuses:
            return 'Overuse'
        
        # Jika ada underuse
        if 'Underuse' in statuses:
            return 'Underuse'
        
        return 'Normal'
    
    df['Final_Status'] = df.apply(get_overall_status, axis=1)
    
    return df

def preprocess_pipeline(df: pd.DataFrame, standards_manager=None) -> pd.DataFrame:
    """
    Pipeline lengkap preprocessing
    
    Args:
        df: Raw DataFrame
        standards_manager: Optional StandardsManager instance untuk labeling
    """
    df = clean_data(df)
    df = calculate_totals(df)
    df = calculate_per_ha(df)
    
    # Apply standards labeling jika ada
    if standards_manager:
        df = apply_standards_labeling(df, standards_manager)
    
    return df
