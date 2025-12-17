"""
Anomaly Explanation Module - NEW APPROACH
Menjelaskan anomali berdasarkan total pupuk per ha, TANPA menyebut MT
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional

def calculate_feature_contributions(
    row_data: pd.Series,
    komoditas: str,
    df_komoditas: pd.DataFrame
) -> Dict:
    """
    Hitung deviasi dari median komoditas untuk 3 fitur utama
    
    ❌ TIDAK menyebut MT sama sekali
    ✅ Fokus pada total per jenis pupuk per ha
    
    Args:
        row_data: Data petani
        komoditas: Nama komoditas
        df_komoditas: Data semua petani dengan komoditas sama
    
    Returns:
        Dict dengan kontribusi per fitur
    """
    pupuk_types = ['Urea', 'NPK', 'Organik']
    contributions = {}
    
    for pupuk in pupuk_types:
        per_ha_col = f'{pupuk}_per_ha'
        
        if per_ha_col not in row_data or per_ha_col not in df_komoditas.columns:
            continue
        
        value = row_data[per_ha_col]
        median = df_komoditas[per_ha_col].median()
        
        # Hitung deviasi
        if median > 0:
            deviation_pct = ((value - median) / median) * 100
        else:
            deviation_pct = 0
        
        contributions[pupuk] = {
            'value': value,
            'median': median,
            'deviation_pct': deviation_pct,
            'direction': 'lebih tinggi' if value > median else 'lebih rendah'
        }
    
    return contributions

def generate_human_explanation(
    petani_name: str,
    komoditas: str,
    contributions: Dict,
    anomaly_label: str,
    standards_manager = None
) -> str:
    """
    Generate penjelasan human-readable TANPA menyebut MT
    
    UPDATED: Standar adalah BATAS ATAS hak, bukan target ideal
    
    Fokus pada:
    - Apa yang normal (median komoditas)
    - Petani ini bagaimana
    - Di bagian mana berbeda (JENIS PUPUK, bukan MT)
    - CONCRETE NUMBERS dari standar jika tersedia
    """
    explanation = []
    
    # Header
    if anomaly_label == 'Anomali':
        explanation.append(f"**Petani {petani_name}** terdeteksi memiliki pola penggunaan pupuk yang berbeda dari mayoritas petani {komoditas} lainnya.\n")
    else:
        explanation.append(f"**Petani {petani_name}** memiliki pola penggunaan pupuk yang normal untuk {komoditas}.\n")
    
    explanation.append("**Detail Penggunaan Pupuk per Hektar:**\n")
    
    standard = None
    if standards_manager:
        standard = standards_manager.get_standard(komoditas)
    
    for pupuk, data in contributions.items():
        value = data['value']
        median = data['median']
        deviation_pct = data['deviation_pct']
        direction = data['direction']
        
        standard_info = ""
        if standard and pupuk in standard:
            std_max = standard[pupuk]['max']  # Only use max, not min
            
            if value > std_max:
                overuse_amount = value - std_max
                overuse_pct = (overuse_amount / std_max) * 100
                standard_info = f" | **Batas Maksimal {komoditas}**: {std_max:.0f} kg/ha → ⚠️ OVERUSE (+{overuse_amount:.1f} kg atau +{overuse_pct:.1f}% melebihi batas)"
            else:
                remaining = std_max - value
                usage_pct = (value / std_max) * 100
                standard_info = f" | **Batas Maksimal {komoditas}**: {std_max:.0f} kg/ha → ✅ Dalam batas ({usage_pct:.0f}% dari maksimal, sisa hak: {remaining:.1f} kg)"
        
        # Format penjelasan
        if abs(deviation_pct) > 20:  # Signifikan berbeda
            explanation.append(
                f"- **{pupuk}**: {value:.1f} kg/ha "
                f"({direction} {abs(deviation_pct):.0f}% dari median {komoditas}: {median:.1f} kg/ha) "
                f"{standard_info}\n"
            )
        else:
            explanation.append(
                f"- **{pupuk}**: {value:.1f} kg/ha "
                f"(mendekati median {komoditas}: {median:.1f} kg/ha) "
                f"{standard_info}\n"
            )
    
    # Kesimpulan
    explanation.append("\n**Interpretasi:**\n")
    
    explanation.append(
        "📌 **Catatan Penting**: Standar pupuk adalah **batas atas hak/alokasi maksimal** yang boleh ditebus petani, "
        "BUKAN target ideal. Petani boleh menggunakan lebih rendah dari standar tanpa masalah.\n\n"
    )
    
    # Cari pupuk dengan deviasi terbesar
    max_deviation_pupuk = max(contributions.keys(), 
                             key=lambda p: abs(contributions[p]['deviation_pct']))
    max_deviation = contributions[max_deviation_pupuk]['deviation_pct']
    
    if anomaly_label == 'Anomali':
        if max_deviation > 0:
            explanation.append(
                f"Petani ini menggunakan **{max_deviation_pupuk}** jauh lebih tinggi dari mayoritas petani {komoditas}. "
            )
            if standard and max_deviation_pupuk in standard:
                std_max = standard[max_deviation_pupuk]['max']
                actual = contributions[max_deviation_pupuk]['value']
                if actual > std_max:
                    explanation.append(
                        f"**⚠️ OVERUSE TERDETEKSI**: Batas maksimal {komoditas} untuk {max_deviation_pupuk}: **{std_max} kg/ha**, "
                        f"petani menggunakan: **{actual:.1f} kg/ha** (melebihi **+{actual - std_max:.1f} kg/ha**). "
                        f"Ini berarti melebihi alokasi hak subsidi.\n"
                    )
                else:
                    explanation.append(
                        f"Meskipun lebih tinggi dari median, masih dalam batas maksimal ({std_max} kg/ha). "
                        f"Perlu evaluasi apakah penggunaan ini efisien.\n"
                    )
            else:
                explanation.append(
                    f"Perlu dievaluasi apakah penggunaan ini sesuai dengan kondisi lahan.\n"
                )
        else:
            explanation.append(
                f"Petani ini menggunakan **{max_deviation_pupuk}** jauh lebih rendah dari mayoritas petani {komoditas}. "
            )
            explanation.append(
                f"**ℹ️ UNDERUSE STATISTIK**: Penggunaan sangat di bawah pola umum (bukan pelanggaran). "
                f"Bisa jadi kondisi lahan berbeda atau ada kendala akses pupuk.\n"
            )
    else:
        explanation.append(
            f"Penggunaan pupuk petani ini konsisten dengan pola umum petani {komoditas} lainnya.\n"
        )
    
    return ''.join(explanation)

def get_anomaly_explanation(
    id_petani: str,
    df: pd.DataFrame,
    standards_manager = None  # Added standards_manager parameter
) -> Dict:
    """
    Main function untuk mendapatkan penjelasan lengkap anomaly
    
    Args:
        id_petani: ID petani yang akan dijelaskan
        df: DataFrame lengkap dengan semua data (sudah ada hasil anomaly detection)
        standards_manager: StandardsManager instance for concrete standard numbers
    
    Returns:
        Dict berisi explanation lengkap
    """
    # 1. Ambil row data berdasarkan ID
    if 'ID_Petani' in df.columns:
        row_data = df[df['ID_Petani'] == id_petani].iloc[0]
    elif 'ID' in df.columns:
        row_data = df[df['ID'] == id_petani].iloc[0]
    else:
        # Jika tidak ada kolom ID, coba convert ke int untuk index
        try:
            row_data = df.iloc[int(id_petani)]
        except (ValueError, IndexError):
            raise ValueError(f"Cannot find petani with ID: {id_petani}")
    
    # 2. Ambil data komoditas
    komoditas = row_data.get('Komoditas', 'Unknown')
    df_komoditas = df[df['Komoditas'] == komoditas]
    
    # 3. Ambil anomaly label
    anomaly_label = row_data.get('Anomaly_Label', 'Normal')
    anomaly_score = row_data.get('Anomaly_Score', 0.0)
    
    # 4. Hitung kontribusi
    contributions = calculate_feature_contributions(row_data, komoditas, df_komoditas)
    
    # 5. Generate penjelasan
    petani_name = row_data.get('ID_Petani', id_petani)
    explanation_text = generate_human_explanation(
        petani_name, komoditas, contributions, anomaly_label, standards_manager  # Pass standards_manager
    )
    
    # 6. Buat result
    result = {
        'id_petani': petani_name,
        'komoditas': komoditas,
        'anomaly_label': anomaly_label,
        'anomaly_score': float(anomaly_score),
        'contributions': contributions,
        'explanation_text': explanation_text
    }
    
    return result

def get_anomaly_comparison(
    df: pd.DataFrame,
    id_petani: str,
    n_similar: int = 5
) -> pd.DataFrame:
    """
    Bandingkan petani anomali dengan petani serupa (normal) dalam komoditas sama
    
    Args:
        df: DataFrame lengkap
        id_petani: ID petani yang ingin dibandingkan
        n_similar: Jumlah petani serupa yang ditampilkan
    
    Returns:
        DataFrame berisi perbandingan
    """
    # Ambil row petani
    if 'ID_Petani' in df.columns:
        target_row = df[df['ID_Petani'] == id_petani].iloc[0]
        row_index = df[df['ID_Petani'] == id_petani].index[0]
    else:
        row_index = int(id_petani)
        target_row = df.iloc[row_index]
    
    komoditas = target_row['Komoditas']
    
    # Filter petani dengan komoditas sama
    same_commodity = df[df['Komoditas'] == komoditas].copy()
    
    # Hitung euclidean distance berdasarkan 3 fitur utama
    feature_names = ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha']
    feature_names = [f for f in feature_names if f in same_commodity.columns]
    
    distances = []
    for idx, row in same_commodity.iterrows():
        if idx == row_index:
            continue
        
        dist = 0
        for feature in feature_names:
            dist += (row[feature] - target_row[feature]) ** 2
        
        distances.append({
            'index': idx,
            'distance': np.sqrt(dist),
            'anomaly_label': row.get('Anomaly_Label', 'unknown')
        })
    
    # Sort by distance dan ambil n_similar terdekat
    distances.sort(key=lambda x: x['distance'])
    similar_indices = [d['index'] for d in distances[:n_similar]]
    
    # Buat comparison dataframe
    comparison_df = same_commodity.loc[similar_indices].copy()
    
    # Tambahkan target row di atas
    target_df = pd.DataFrame([target_row])
    target_df['comparison_type'] = 'Target (You)'
    comparison_df['comparison_type'] = 'Similar Farmers'
    
    result = pd.concat([target_df, comparison_df], ignore_index=True)
    
    # Select kolom yang relevan
    display_cols = ['ID_Petani', 'Komoditas', 'Urea_per_ha', 'NPK_per_ha', 
                   'Organik_per_ha', 'Anomaly_Label', 'comparison_type']
    display_cols = [c for c in display_cols if c in result.columns]
    
    return result[display_cols]

def export_explanation_to_json(explanation: Dict, output_path: str):
    """
    Export penjelasan ke file JSON
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(explanation, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Penjelasan tersimpan: {output_path}")

def calculate_median_and_std(df: pd.DataFrame, komoditas: str = None) -> Dict:
    """
    Calculate median and std for each fertilizer type per ha
    Used for comparison purposes
    
    Args:
        df: DataFrame with fertilizer data
        komoditas: Optional komoditas filter
    
    Returns:
        Dict with median and std values
    """
    if komoditas:
        df_filtered = df[df['Komoditas'] == komoditas]
    else:
        df_filtered = df
    
    pupuk_types = ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha']
    stats = {}
    
    for pupuk in pupuk_types:
        if pupuk in df_filtered.columns:
            stats[pupuk] = {
                'median': df_filtered[pupuk].median(),
                'std': df_filtered[pupuk].std(),
                'mean': df_filtered[pupuk].mean(),
                'q25': df_filtered[pupuk].quantile(0.25),
                'q75': df_filtered[pupuk].quantile(0.75)
            }
    
    return stats
