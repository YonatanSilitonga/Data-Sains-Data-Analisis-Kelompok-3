"""
Utility Functions Module
Fungsi-fungsi pembantu untuk sistem RDKK
UPDATED: TIDAK ADA min/max statis - semua dinamis dari distribusi data atau standar
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def safe_divide(numerator, denominator, fill_value=0):
    """Pembagian aman dengan handling divide by zero"""
    denominator_safe = np.where(denominator == 0, np.nan, denominator)
    result = numerator / denominator_safe
    return np.nan_to_num(result, nan=fill_value)

def normalize_features(df, columns, method='standard'):
    """
    Normalisasi fitur numerik
    
    Parameters:
    - df: DataFrame
    - columns: list kolom yang akan dinormalisasi
    - method: 'standard' (z-score) atau 'minmax' (0-1)
    """
    df_copy = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method harus 'standard' atau 'minmax'")
    
    existing_cols = [col for col in columns if col in df_copy.columns]
    
    if existing_cols:
        df_copy[existing_cols] = scaler.fit_transform(df_copy[existing_cols])
    
    return df_copy, scaler

def get_total_pupuk_columns(df):
    """
    Ambil kolom total pupuk per jenis per ha (konsep baru)
    TIDAK menggunakan MT1/MT2/MT3 individual
    """
    base_cols = ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha', 'Total_per_ha']
    return [col for col in base_cols if col in df.columns]

def get_feature_columns(df, exclude_cols=None):
    """
    Ambil kolom fitur untuk ML (exclude ID, kategori, dll)
    
    Parameters:
    - df: DataFrame
    - exclude_cols: list kolom yang tidak digunakan untuk ML
    """
    if exclude_cols is None:
        exclude_cols = ['ID_Petani', 'Desa', 'Kelompok_Tani', 'Komoditas']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    return feature_cols

def create_summary_stats(df, group_by='Komoditas'):
    """Buat ringkasan statistik per grup"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    summary = df.groupby(group_by)[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max'])
    
    return summary

def detect_outliers_iqr(df, column):
    """Deteksi outlier menggunakan IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return outliers, lower_bound, upper_bound

def calculate_anomaly_score(values, mean, std):
    """
    Hitung skor anomali berdasarkan jarak dari mean
    Skor 0-1, dimana 1 = sangat anomali
    """
    if std == 0:
        return np.zeros_like(values)
    
    z_scores = np.abs((values - mean) / std)
    # Normalisasi ke 0-1 menggunakan sigmoid-like function
    scores = 1 - np.exp(-z_scores / 2)
    
    return scores

def categorize_anomaly_severity(scores, threshold_low=0.3, threshold_high=0.7):
    """
    Kategorikan tingkat keparahan anomali
    
    Returns:
    - 'Normal': skor < threshold_low
    - 'Anomali Ringan': threshold_low <= skor < threshold_high
    - 'Anomali Berat': skor >= threshold_high
    """
    categories = []
    for score in scores:
        if score < threshold_low:
            categories.append('Normal')
        elif score < threshold_high:
            categories.append('Anomali Ringan')
        else:
            categories.append('Anomali Berat')
    
    return categories

def format_number(value, decimals=2):
    """Format angka dengan pemisah ribuan"""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"

def calculate_median_standards(df, komoditas_col='Komoditas'):
    """
    Hitung median pupuk per komoditas sebagai baseline standar dinamis
    UPDATED: Gunakan total per ha per jenis, BUKAN per MT
    
    Returns:
        dict: {komoditas: {pupuk_col: {'median': value, 'q1': value, 'q3': value}}}
    """
    median_standards = {}
    
    pupuk_cols = ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha', 'Total_per_ha']
    
    for komoditas in df[komoditas_col].unique():
        komoditas_data = df[df[komoditas_col] == komoditas]
        median_standards[komoditas] = {}
        
        for col in pupuk_cols:
            if col in komoditas_data.columns:
                median_val = komoditas_data[col].median()
                q1_val = komoditas_data[col].quantile(0.25)
                q3_val = komoditas_data[col].quantile(0.75)
                
                median_standards[komoditas][col] = {
                    'median': median_val,
                    'q1': q1_val,
                    'q3': q3_val,
                    'tolerance_lower': median_val * 0.7,  # 30% di bawah median
                    'tolerance_upper': median_val * 1.3   # 30% di atas median
                }
    
    return median_standards

def compare_with_median(value, median_std, tolerance=0.3):
    """
    Bandingkan nilai dengan median ± toleransi
    
    Parameters:
    - value: nilai aktual
    - median_std: dict dengan 'median', 'tolerance_lower', 'tolerance_upper'
    - tolerance: toleransi persentase (default 0.3 = 30%)
    
    Returns:
    - 'Normal': dalam rentang toleransi
    - 'Underuse': < median - 30%
    - 'Overuse': > median + 30%
    """
    if pd.isna(value) or not median_std or pd.isna(median_std.get('median')):
        return 'Unknown'
    
    median_val = median_std['median']
    
    if median_val == 0:
        return 'Unknown'
    
    lower_bound = median_std.get('tolerance_lower', median_val * (1 - tolerance))
    upper_bound = median_std.get('tolerance_upper', median_val * (1 + tolerance))
    
    if value < lower_bound:
        return 'Underuse'
    elif value > upper_bound:
        return 'Overuse'
    else:
        return 'Normal'

def compare_with_standard(value, standard, tolerance=0.2):
    """
    Bandingkan nilai dengan standar
    
    UPDATED: Hanya cek overuse (melebihi max), bukan underuse
    
    Parameters:
    - value: nilai aktual
    - standard: nilai standar (dict dengan 'max' key atau float)
    - tolerance: toleransi (0.2 = 20%)
    
    Returns:
    - 'Normal': tidak melebihi max
    - 'Overuse': melebihi max
    - 'Tidak Ada Standar': jika standar tidak valid
    """
    if isinstance(standard, dict):
        if 'max' in standard:
            if isinstance(standard['max'], (int, float)):
                standard_max = standard['max']
            else:
                return 'Tidak Ada Standar'
        else:
            return 'Tidak Ada Standar'
    else:
        standard_max = standard
    
    if pd.isna(value) or pd.isna(standard_max) or standard_max == 0:
        return 'Tidak Ada Standar'
    
    upper = standard_max * (1 + tolerance)
    
    if value > upper:
        return 'Overuse'
    else:
        return 'Normal'

def calculate_percentage_difference(value, reference):
    """Hitung persentase perbedaan"""
    if reference == 0 or pd.isna(reference):
        return np.nan
    
    return ((value - reference) / reference) * 100

def extract_numeric_features(df):
    """Extract hanya fitur numerik untuk ML"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Hapus kolom dengan variance = 0
    variance = numeric_df.var()
    valid_cols = variance[variance > 0].index.tolist()
    
    return numeric_df[valid_cols]

def handle_missing_values(df, strategy='median'):
    """
    Handle missing values
    
    Parameters:
    - strategy: 'median', 'mean', atau 'zero'
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df_copy[col].isna().sum() > 0:
            if strategy == 'median':
                fill_value = df_copy[col].median()
            elif strategy == 'mean':
                fill_value = df_copy[col].mean()
            else:
                fill_value = 0
            
            df_copy[col] = df_copy[col].fillna(fill_value)
    
    return df_copy

def print_section_header(title):
    """Print header section yang rapi"""
    print("\n" + "="*60)
    print(title.center(60))
    print("="*60 + "\n")

def print_info(message, symbol="→"):
    """Print informasi dengan simbol"""
    print(f"{symbol} {message}")

def print_success(message):
    """Print pesan sukses"""
    print(f"✓ {message}")

def print_error(message):
    """Print pesan error"""
    print(f"✗ {message}")

if __name__ == "__main__":
    print("Utils module loaded successfully!")
    print("\nAvailable functions:")
    print("  - safe_divide()")
    print("  - normalize_features()")
    print("  - calculate_median_standards() [UPDATED]")
    print("  - compare_with_median()")
    print("  - categorize_anomaly_severity()")
    print("  ... and more!")
