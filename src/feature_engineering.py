import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path if running as standalone
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))

def calculate_total_pupuk_per_ha(df):
    """
    FITUR UTAMA BARU: Hitung total pupuk per jenis per hektar
    Formula: (Total_MT1 + Total_MT2 + Total_MT3) / Luas_ha
    
    INI ADALAH SATU-SATUNYA FITUR YANG DIGUNAKAN UNTUK DETEKSI ANOMALI.
    MT tidak lagi menjadi basis penilaian anomali.
    """
    df_copy = df.copy()
    
    pupuk_types = ['Urea', 'NPK', 'Organik']
    luas_col = 'Luas_Tanah_ha' if 'Luas_Tanah_ha' in df_copy.columns else 'Luas_ha'
    
    for pupuk in pupuk_types:
        mt_cols = [f'{pupuk}_MT1', f'{pupuk}_MT2', f'{pupuk}_MT3']
        existing_mt = [col for col in mt_cols if col in df_copy.columns]
        
        if existing_mt and luas_col in df_copy.columns:
            total_col = f'Total_{pupuk}'
            if total_col not in df_copy.columns:
                df_copy[total_col] = df_copy[existing_mt].sum(axis=1)
            
            df_copy[f'{pupuk}_per_ha'] = df_copy[total_col] / df_copy[luas_col].replace(0, np.nan)
            df_copy[f'{pupuk}_per_ha'] = df_copy[f'{pupuk}_per_ha'].fillna(0)
    
    if all(f'{p}_per_ha' in df_copy.columns for p in pupuk_types):
        df_copy['Total_per_ha'] = (
            df_copy['Urea_per_ha'] + 
            df_copy['NPK_per_ha'] + 
            df_copy['Organik_per_ha']
        )
        print(f"✓ Total_per_ha berhasil dibuat untuk visualisasi dashboard")
    
    print(f"✓ Total pupuk per ha dihitung (Urea, NPK, Organik per ha)")
    return df_copy

def calculate_total_per_mt(df):
    """
    Hitung total pupuk per MT
    CATATAN: Ini untuk keperluan clustering/analisis perilaku, BUKAN untuk anomali detection
    """
    df_copy = df.copy()
    
    pupuk_types = ['Urea', 'NPK', 'Organik']
    mt_periods = ['MT1', 'MT2', 'MT3']
    
    for mt in mt_periods:
        cols = [f'{pupuk}_{mt}' for pupuk in pupuk_types if f'{pupuk}_{mt}' in df_copy.columns]
        if cols:
            df_copy[f'Total_{mt}'] = df_copy[cols].sum(axis=1)
    
    print(f"✓ Total pupuk per MT dihitung (untuk clustering, bukan anomali)")
    return df_copy

def calculate_median_labels(df):
    """
    CRITICAL CHANGE: Label anomaly berdasarkan MEDIAN per komoditas (IQR method)
    
    Anomali = penggunaan total pupuk per ha yang berbeda signifikan dari mayoritas
    Acuan: median + distribusi aktual (TIDAK ada min/max statis)
    """
    df_copy = df.copy()
    
    pupuk_types = ['Urea', 'NPK', 'Organik']
    
    for pupuk in pupuk_types:
        per_ha_col = f'{pupuk}_per_ha'
        label_col = f'{pupuk}_Label'
        
        if per_ha_col not in df_copy.columns:
            continue
        
        df_copy[label_col] = 'Normal'
        
        for komoditas in df_copy['Komoditas'].unique():
            mask = df_copy['Komoditas'] == komoditas
            komoditas_data = df_copy.loc[mask, per_ha_col]
            
            if len(komoditas_data) == 0:
                continue
            
            median_val = komoditas_data.median()
            
            if median_val == 0:
                continue
            
            # Menggunakan distribusi aktual data (IQR method)
            Q1 = komoditas_data.quantile(0.25)
            Q3 = komoditas_data.quantile(0.75)
            IQR = Q3 - Q1
            
            # Batas anomali: Q1 - 1.5*IQR dan Q3 + 1.5*IQR
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Assign labels
            underuse_mask = mask & (df_copy[per_ha_col] < lower_bound)
            overuse_mask = mask & (df_copy[per_ha_col] > upper_bound)
            
            df_copy.loc[underuse_mask, label_col] = 'Underuse'
            df_copy.loc[overuse_mask, label_col] = 'Overuse'
            
            # Store bounds untuk referensi
            df_copy.loc[mask, f'{pupuk}_Median'] = median_val
            df_copy.loc[mask, f'{pupuk}_Lower_Bound'] = lower_bound
            df_copy.loc[mask, f'{pupuk}_Upper_Bound'] = upper_bound
    
    def get_overall_status(row):
        labels = [row.get(f'{p}_Label', 'Normal') for p in pupuk_types]
        
        # Prioritas: Overuse > Underuse > Normal
        if 'Overuse' in labels:
            return 'Overuse'
        elif 'Underuse' in labels:
            return 'Underuse'
        else:
            return 'Normal'
    
    df_copy['Overall_Status'] = df_copy.apply(get_overall_status, axis=1)
    
    print(f"✓ Median-based labels dihitung (IQR method, bukan MT)")
    return df_copy

def feature_engineering_pipeline(df, standards_dict=None):
    """
    Pipeline feature engineering BARU - fokus pada total pupuk per ha
    
    HANYA menghasilkan 3 fitur utama untuk anomali:
    - Urea_per_ha
    - NPK_per_ha
    - Organik_per_ha
    (+ opsional: Luas_Lahan untuk konteks)
    
    MT TIDAK digunakan untuk anomali detection.
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING - NEW APPROACH")
    print("Fokus: Total Pupuk per Jenis per Hektar")
    print("TIDAK menggunakan MT sebagai basis anomali")
    print("="*60 + "\n")
    
    df_features = df.copy()
    
    df_features = calculate_total_pupuk_per_ha(df_features)
    
    # 2. Total per MT (untuk clustering/analisis, bukan anomali)
    df_features = calculate_total_per_mt(df_features)
    
    df_features = calculate_median_labels(df_features)
    
    print("\n" + "="*60)
    print(f"FEATURE ENGINEERING SELESAI")
    print(f"Fitur utama untuk anomali detection:")
    print(f"  - Urea_per_ha (TOTAL Urea / Luas)")
    print(f"  - NPK_per_ha (TOTAL NPK / Luas)")
    print(f"  - Organik_per_ha (TOTAL Organik / Luas)")
    print(f"\n❌ MT1, MT2, MT3 TIDAK digunakan untuk anomali")
    print("="*60 + "\n")
    
    return df_features

def save_features(df, output_path):
    """Simpan dataset dengan fitur"""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Dataset dengan fitur tersimpan: {output_path}")
    except Exception as e:
        print(f"✗ Error saving features: {e}")

if __name__ == "__main__":
    # Test feature engineering
    from src.data_processing import preprocess_pipeline
    
    # Preprocessing dulu
    input_path = "data/rdkk_dataset.csv"
    cleaned_path = "output/dataset_cleaned.csv"
    df_clean = preprocess_pipeline(input_path, cleaned_path)
    
    if df_clean is not None:
        # Feature engineering
        df_features = feature_engineering_pipeline(df_clean)
        
        # Simpan
        output_path = "output/dataset_with_features.csv"
        save_features(df_features, output_path)
        
        print("\nSample fitur baru:")
        key_cols = ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha', 'Overall_Status']
        existing_key_cols = [col for col in key_cols if col in df_features.columns]
        print(df_features[existing_key_cols].head())
