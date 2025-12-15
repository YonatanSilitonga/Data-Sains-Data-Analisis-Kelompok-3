"""
Data Processing Module
Modul untuk preprocessing dataset RDKK
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_data(filepath):
    """Load CSV dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

def replace_zeros_with_nan(df, pupuk_columns):
    """Ganti nilai 0 dengan NaN untuk kolom pupuk"""
    df_copy = df.copy()
    for col in pupuk_columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].replace(0, np.nan)
    print(f"✓ Nilai 0 diganti dengan NaN untuk {len(pupuk_columns)} kolom pupuk")
    return df_copy

def impute_by_komoditas(df, pupuk_columns):
    """Imputasi median per komoditas untuk kolom pupuk"""
    df_copy = df.copy()
    
    for col in pupuk_columns:
        if col in df_copy.columns:
            # Hitung median per komoditas
            median_values = df_copy.groupby('Komoditas')[col].transform('median')
            # Isi NaN dengan median komoditas
            df_copy[col] = df_copy[col].fillna(median_values)
            # Jika masih ada NaN (komoditas tanpa data), isi dengan 0
            df_copy[col] = df_copy[col].fillna(0)
    
    print(f"✓ Imputasi median per komoditas selesai untuk {len(pupuk_columns)} kolom")
    return df_copy

def create_petani_id(df):
    """Buat ID_Petani unik jika belum ada"""
    if 'ID_Petani' not in df.columns:
        df['ID_Petani'] = ['P' + str(i+1).zfill(4) for i in range(len(df))]
        print(f"✓ ID_Petani dibuat: P0001 - P{len(df):04d}")
    return df

def rename_standardize_columns(df):
    """Rename dan standardisasi nama kolom"""
    column_mapping = {
        'Luas Tanah (m2)': 'Luas_Tanah_m2',
        'Luas Tanah': 'Luas_Tanah_m2',
        'Kelompok Tani': 'Kelompok_Tani'
    }
    
    df_renamed = df.rename(columns=column_mapping)
    print(f"✓ Kolom distandarisasi")
    return df_renamed

def reorder_columns(df):
    """Urutkan kolom sesuai format RDKK"""
    priority_cols = ['ID_Petani', 'Desa', 'Kelompok_Tani', 'Komoditas', 'Luas_Tanah_m2']
    
    # Pupuk MT1, MT2, MT3
    pupuk_cols = []
    for mt in ['MT1', 'MT2', 'MT3']:
        for jenis in ['Urea', 'NPK', 'Organik']:
            col = f'{jenis}_{mt}'
            if col in df.columns:
                pupuk_cols.append(col)
    
    # Kolom lainnya
    other_cols = [col for col in df.columns if col not in priority_cols + pupuk_cols]
    
    # Urutkan ulang
    new_order = priority_cols + pupuk_cols + other_cols
    new_order = [col for col in new_order if col in df.columns]
    
    df_reordered = df[new_order]
    print(f"✓ Kolom diurutkan ulang")
    return df_reordered

def save_processed_data(df, output_path):
    """Simpan dataset yang sudah diproses"""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Dataset tersimpan: {output_path}")
    except Exception as e:
        print(f"✗ Error saving data: {e}")

def preprocess_pipeline(input_filepath, output_filepath):
    """Pipeline lengkap preprocessing"""
    print("\n" + "="*60)
    print("PREPROCESSING DATA RDKK")
    print("="*60 + "\n")
    
    # 1. Load data
    df = load_data(input_filepath)
    if df is None:
        return None
    
    # 2. Identifikasi kolom pupuk
    pupuk_columns = [col for col in df.columns if any(x in col for x in ['Urea', 'NPK', 'Organik']) 
                     and any(mt in col for mt in ['MT1', 'MT2', 'MT3'])]
    print(f"\n→ Kolom pupuk teridentifikasi: {len(pupuk_columns)}")
    
    # 3. Replace 0 dengan NaN
    df = replace_zeros_with_nan(df, pupuk_columns)
    
    # 4. Imputasi median per komoditas
    df = impute_by_komoditas(df, pupuk_columns)
    
    # 5. Buat ID Petani
    df = create_petani_id(df)
    
    # 6. Rename kolom
    df = rename_standardize_columns(df)
    
    # 7. Reorder kolom
    df = reorder_columns(df)
    
    # 8. Konversi Luas ke hektar (ha)
    if 'Luas_Tanah_m2' in df.columns:
        df['Luas_ha'] = df['Luas_Tanah_m2'] / 10000
        print(f"✓ Luas tanah dikonversi ke hektar")
    
    # 9. Simpan
    save_processed_data(df, output_filepath)
    
    print("\n" + "="*60)
    print(f"PREPROCESSING SELESAI")
    print(f"Total data: {len(df)} petani")
    print(f"Total kolom: {len(df.columns)}")
    print("="*60 + "\n")
    
    return df

if __name__ == "__main__":
    # Test preprocessing
    input_path = "data/rdkk_dataset.csv"
    output_path = "output/dataset_cleaned.csv"
    
    df_clean = preprocess_pipeline(input_path, output_path)
    
    if df_clean is not None:
        print("\nSample data:")
        print(df_clean.head())
        print("\nInfo dataset:")
        print(df_clean.info())
