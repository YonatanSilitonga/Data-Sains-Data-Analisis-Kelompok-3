"""
Recommendation Module
Generate rekomendasi berdasarkan anomali, cluster, dan standar pupuk (opsional)
Terintegrasi dengan median-based analysis
"""

import pandas as pd
import numpy as np
from typing import Optional

def generate_recommendations(df: pd.DataFrame, standards_enabled: bool = True, include_details: bool = True) -> pd.DataFrame:
    """
    Generate rekomendasi untuk setiap petani
    Gabungan dari: ML anomaly + standar pupuk (jika aktif) + pola cluster
    
    Args:
        df: DataFrame dengan hasil preprocessing, anomaly, dan clustering
        standards_enabled: Apakah standar pupuk digunakan
        include_details: Include detail analisis
    
    Returns:
        DataFrame dengan kolom Rekomendasi, Prioritas, Action_Plan
    """
    df_result = df.copy()
    recommendations = []
    priorities = []
    action_plans = []
    
    for idx, row in df_result.iterrows():
        rec_parts = []
        priority = 'Rendah'
        actions = []
        
        if 'Anomaly_Label' in row.index and row['Anomaly_Label'] == 'Anomali':
            severity = row.get('Anomaly_Severity', 'Unknown')
            score = row.get('Anomaly_Score', 0)
            
            if severity == 'Anomali Berat' or score > 0.7:
                rec_parts.append(f"🚨 ANOMALI BERAT: Pola pupuk sangat tidak wajar (ML Score: {score:.2f})")
                actions.append("URGENT: Investigasi segera! Cek validitas data dan pola pemupukan")
                priority = 'Tinggi'
            elif severity == 'Anomali Ringan' or score > 0.5:
                rec_parts.append(f"⚠️ ANOMALI RINGAN: Pola perlu perhatian (ML Score: {score:.2f})")
                actions.append("Periksa pola pemupukan, ada ketidakwajaran")
                if priority == 'Rendah':
                    priority = 'Sedang'
        
        if 'Median_Status' in row.index:
            median_status = row['Median_Status']
            
            if median_status == 'Overuse':
                rec_parts.append("📈 OVERUSE: Penggunaan melebihi median +30%")
                
                # Detail per pupuk
                for pupuk in ['Urea', 'NPK', 'Organik']:
                    label_col = f'Median_Label_{pupuk}'
                    per_ha_col = f'{pupuk}_per_ha'
                    
                    if label_col in row.index and row[label_col] == 'Overuse':
                        actual = row.get(per_ha_col, 0)
                        rec_parts.append(f"  • {pupuk}: {actual:.1f} kg/ha (>median+30%)")
                
                actions.append("Kurangi dosis pupuk untuk menghindari pemborosan")
                if priority != 'Tinggi':
                    priority = 'Sedang'
            
            elif median_status == 'Underuse':
                rec_parts.append("📉 UNDERUSE: Penggunaan di bawah median -30%")
                
                # Detail per pupuk
                for pupuk in ['Urea', 'NPK', 'Organik']:
                    label_col = f'Median_Label_{pupuk}'
                    per_ha_col = f'{pupuk}_per_ha'
                    
                    if label_col in row.index and row[label_col] == 'Underuse':
                        actual = row.get(per_ha_col, 0)
                        rec_parts.append(f"  • {pupuk}: {actual:.1f} kg/ha (<median-30%)")
                
                actions.append("Pertimbangkan menambah pupuk untuk hasil optimal")
                if priority == 'Rendah':
                    priority = 'Sedang'
        
        if standards_enabled and 'Final_Status' in row.index:
            final_status = row['Final_Status']
            
            if final_status == 'Overuse' and 'Median_Status' not in row.index:
                rec_parts.append("⚠️ Melebihi standar admin")
                actions.append("Sesuaikan dengan standar yang ditetapkan")
                if priority != 'Tinggi' and priority != 'Sedang':
                    priority = 'Sedang'
            
            elif final_status == 'Underuse' and 'Median_Status' not in row.index:
                rec_parts.append("📉 Di bawah standar admin")
                actions.append("Tambahkan pupuk sesuai standar")
                if priority == 'Rendah':
                    priority = 'Sedang'
        
        # 4. Intensitas validation
        if 'Total_per_ha' in row.index:
            intensitas = row['Total_per_ha']
            
            if intensitas > 1200:
                rec_parts.append(f"📊 Intensitas sangat tinggi: {intensitas:.1f} kg/ha")
                actions.append("Evaluasi efisiensi, kemungkinan boros")
                if priority == 'Rendah':
                    priority = 'Sedang'
            elif intensitas < 150:
                rec_parts.append(f"📊 Intensitas rendah: {intensitas:.1f} kg/ha")
                actions.append("Pastikan nutrisi tanaman terpenuhi")
        
        # 5. Cluster insights
        if 'Cluster_Label' in row.index:
            cluster = row['Cluster_Label']
            rec_parts.append(f"🎯 {cluster}")
        
        # 6. Luas lahan validation
        if 'Luas_Tanah_ha' in row.index:
            luas = row['Luas_Tanah_ha']
            
            if luas < 0.1:
                rec_parts.append("⚠️ Luas sangat kecil - verifikasi pengukuran")
                actions.append("Cek kembali data luas lahan")
                if priority == 'Rendah':
                    priority = 'Sedang'
        
        # Compile hasil
        if not rec_parts:
            rec_parts.append("✅ Data normal, pola pemupukan baik")
            actions.append("Lanjutkan praktek saat ini")
        
        if not actions:
            actions.append("Tidak ada tindakan diperlukan")
        
        recommendations.append(" | ".join(rec_parts))
        priorities.append(priority)
        action_plans.append("; ".join(actions))
    
    df_result['Rekomendasi'] = recommendations
    df_result['Prioritas'] = priorities
    df_result['Action_Plan'] = action_plans
    
    return df_result

def export_recommendations_report(df: pd.DataFrame, output_path: str):
    """Export laporan rekomendasi ke CSV"""
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select relevant columns
    export_cols = [
        'ID_Petani', 'Desa', 'Kelompok_Tani', 'Komoditas', 
        'Luas_Tanah_ha', 'Final_Status', 'Prioritas', 
        'Rekomendasi', 'Action_Plan'
    ]
    export_cols = [col for col in export_cols if col in df.columns]
    
    df[export_cols].to_csv(output_path, index=False)
    print(f"✓ Laporan rekomendasi tersimpan: {output_path}")

def print_recommendation_summary(df: pd.DataFrame):
    """Print ringkasan rekomendasi"""
    print("\n" + "="*60)
    print("RINGKASAN REKOMENDASI")
    print("="*60)
    
    total = len(df)
    tinggi = (df['Prioritas'] == 'Tinggi').sum()
    sedang = (df['Prioritas'] == 'Sedang').sum()
    rendah = (df['Prioritas'] == 'Rendah').sum()
    
    print(f"Total petani: {total:,}")
    print(f"  - Prioritas Tinggi: {tinggi:,} ({tinggi/total*100:.1f}%)")
    print(f"  - Prioritas Sedang: {sedang:,} ({sedang/total*100:.1f}%)")
    print(f"  - Prioritas Rendah: {rendah:,} ({rendah/total*100:.1f}%)")
    
    if tinggi > 0:
        print(f"\n⚠️  {tinggi:,} petani memerlukan tindakan segera!")
    
    print("="*60 + "\n")
