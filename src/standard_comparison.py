"""
Standard Comparison Module
Modul untuk membandingkan penggunaan pupuk petani dengan standar
dan menghasilkan penjelasan yang mudah dipahami
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def get_standard_for_commodity(standards_manager, komoditas: str) -> Dict:
    """
    Ambil standar pupuk untuk komoditas tertentu
    
    Returns:
        dict dengan struktur {Urea: {min, max}, NPK: {min, max}, Organik: {min, max}}
    """
    standard = standards_manager.get_standard(komoditas)
    if not standard or not isinstance(standard, dict):
        return None
    return standard

def calculate_standard_comparison(petani_data: pd.Series, standards_manager, komoditas: str) -> Dict:
    """
    Bandingkan penggunaan pupuk petani dengan standar (jika ada)
    
    UPDATED: Standar hanya untuk deteksi OVERUSE (melebihi max)
    Underuse TIDAK menggunakan min standar
    
    Returns:
        dict dengan detail perbandingan untuk setiap jenis pupuk
    """
    standard = get_standard_for_commodity(standards_manager, komoditas)
    
    if not standard:
        return {
            'has_standard': False,
            'message': f'Tidak ada standar pupuk untuk komoditas {komoditas}'
        }
    
    luas_ha = petani_data.get('Luas_Tanah_ha', 0)
    
    if luas_ha == 0:
        return {
            'has_standard': False,
            'message': 'Luas lahan tidak valid'
        }
    
    # Ambil total pupuk dari petani
    total_urea = petani_data.get('Total_Urea', 0)
    total_npk = petani_data.get('Total_NPK', 0)
    total_organik = petani_data.get('Total_Organik', 0)
    
    # Hitung per ha
    urea_per_ha = total_urea / luas_ha if luas_ha > 0 else 0
    npk_per_ha = total_npk / luas_ha if luas_ha > 0 else 0
    organik_per_ha = total_organik / luas_ha if luas_ha > 0 else 0
    
    comparisons = []
    
    for pupuk, per_ha_value in [('Urea', urea_per_ha), ('NPK', npk_per_ha), ('Organik', organik_per_ha)]:
        if pupuk not in standard or not isinstance(standard[pupuk], dict):
            continue
        
        pupuk_std = standard[pupuk]
        std_max = pupuk_std.get('max', 0)
        
        if per_ha_value > std_max:
            overuse_amount = per_ha_value - std_max
            overuse_pct = (overuse_amount / std_max * 100) if std_max > 0 else 0
            
            if overuse_pct <= 10:
                level = 'Overuse Ringan'
            elif overuse_pct <= 30:
                level = 'Overuse Sedang'
            else:
                level = 'Overuse Tinggi'
            status = 'Overuse'
        else:
            usage_pct = (per_ha_value / std_max * 100) if std_max > 0 else 0
            level = 'Normal'
            status = 'Normal'
        
        comparisons.append({
            'jenis': pupuk,
            'standar_max': std_max,
            'petani_per_ha': per_ha_value,
            'selisih_dari_max': per_ha_value - std_max,
            'persen_dari_max': (per_ha_value / std_max * 100) if std_max > 0 else 0,
            'status': status,
            'level': level
        })
    
    return {
        'has_standard': True,
        'komoditas': komoditas,
        'luas_ha': luas_ha,
        'comparisons': comparisons
    }

def generate_explanation_text(comparison_result: Dict) -> List[str]:
    """
    Generate penjelasan dalam bahasa awam tentang overuse
    
    UPDATED: Hanya jelaskan overuse. Underuse handled separately via statistik.
    
    Returns:
        list of string explanations
    """
    if not comparison_result.get('has_standard'):
        return [comparison_result.get('message', 'Tidak ada penjelasan')]
    
    explanations = []
    komoditas = comparison_result.get('komoditas', '')
    luas_ha = comparison_result.get('luas_ha', 0)
    
    explanations.append(f"**Komoditas {komoditas} dengan luas lahan {luas_ha:.2f} ha**")
    explanations.append("")
    explanations.append("**📌 Filosofi Standar**: Standar pupuk adalah **batas atas hak/alokasi maksimal**, BUKAN target ideal. Petani boleh menggunakan lebih rendah tanpa masalah.")
    explanations.append("")
    
    for comp in comparison_result.get('comparisons', []):
        jenis = comp['jenis']
        std_max = comp['standar_max']
        petani = comp['petani_per_ha']
        persen_dari_max = comp['persen_dari_max']
        status = comp['status']
        level = comp['level']
        
        # Header
        if status == 'Overuse':
            emoji = '🔴'
        else:
            emoji = '🟢'
        
        explanations.append(f"{emoji} **{jenis}**: {level}")
        
        explanations.append(f"- Batas Maksimal: {std_max:.0f} kg/ha")
        explanations.append(f"- Petani menggunakan: {petani:.1f} kg/ha ({persen_dari_max:.0f}% dari maksimal)")
        
        if status == 'Overuse':
            overuse_amount = petani - std_max
            explanations.append(f"- ⚠️ **MELEBIHI BATAS**: +{overuse_amount:.1f} kg/ha di atas maksimal")
            if level == 'Overuse Tinggi':
                explanations.append(f"- **Dampak**: Melebihi alokasi hak subsidi. Perlu evaluasi dan penyesuaian.")
            else:
                explanations.append(f"- Sedikit melebihi batas, perlu konfirmasi data.")
        else:
            remaining = std_max - petani
            explanations.append(f"- ✅ Dalam batas (sisa hak: {remaining:.1f} kg/ha)")
        
        explanations.append("")
    
    return explanations

def get_educational_box() -> str:
    """
    Return konten box edukatif tentang standar pupuk
    
    UPDATED: Jelaskan filosofi baru
    """
    return """
    <div class="info-box">
    <strong>📚 Memahami Standar Pupuk:</strong><br>
    <ul style="margin: 10px 0; padding-left: 20px;">
        <li><strong>Standar Pupuk = Batas Atas Hak</strong>: Standar adalah alokasi maksimal (kg/ha) yang BOLEH ditebus petani, 
        BUKAN kewajiban atau target ideal.</li>
        <li><strong>Overuse (Penggunaan Melebihi Batas)</strong>: Menggunakan pupuk melebihi standar maksimal. 
        Ini adalah anomali kebijakan karena melebihi hak alokasi.</li>
        <li><strong>Di Bawah Standar = BOLEH</strong>: Petani boleh membeli pupuk di bawah standar 
        (misal 150 dari 250 kg/ha) tanpa dianggap kesalahan. Ini hak petani untuk menyesuaikan dengan kondisi lahan.</li>
        <li><strong>Underuse Statistik</strong>: Berbeda dari standar, underuse statistik adalah penggunaan yang 
        SANGAT JAUH di bawah mayoritas petani sejenis (bukan pelanggaran kebijakan, tapi pola yang tidak biasa).</li>
        <li>🔍 <em>Catatan</em>: Setiap petani punya kondisi unik. Yang penting: jangan melebihi batas maksimal.</li>
    </ul>
    </div>
    """

def classify_overuse_level(diff_percent: float) -> Tuple[str, str]:
    """
    Klasifikasi tingkat overuse/underuse
    
    Returns:
        (level, color) tuple
    """
    abs_diff = abs(diff_percent)
    
    if abs_diff <= 10:
        return ('Normal', '#4CAF50')
    elif abs_diff <= 30:
        return ('Ringan', '#FF9800')
    else:
        return ('Tinggi', '#F44336')
"""
Standard Comparison Module
Modul untuk membandingkan penggunaan pupuk petani dengan standar
dan menghasilkan penjelasan yang mudah dipahami
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def get_standard_for_commodity(standards_manager, komoditas: str) -> Dict:
    """
    Ambil standar pupuk untuk komoditas tertentu
    
    Returns:
        dict dengan struktur {Urea: {min, max}, NPK: {min, max}, Organik: {min, max}}
    """
    standard = standards_manager.get_standard(komoditas)
    if not standard or not isinstance(standard, dict):
        return None
    return standard

def calculate_standard_comparison(petani_data: pd.Series, standards_manager, komoditas: str) -> Dict:
    """
    Bandingkan penggunaan pupuk petani dengan standar (jika ada)
    
    UPDATED: Standar hanya untuk deteksi OVERUSE (melebihi max)
    Underuse TIDAK menggunakan min standar
    
    Returns:
        dict dengan detail perbandingan untuk setiap jenis pupuk
    """
    standard = get_standard_for_commodity(standards_manager, komoditas)
    
    if not standard:
        return {
            'has_standard': False,
            'message': f'Tidak ada standar pupuk untuk komoditas {komoditas}'
        }
    
    luas_ha = petani_data.get('Luas_Tanah_ha', 0)
    
    if luas_ha == 0:
        return {
            'has_standard': False,
            'message': 'Luas lahan tidak valid'
        }
    
    # Ambil total pupuk dari petani
    total_urea = petani_data.get('Total_Urea', 0)
    total_npk = petani_data.get('Total_NPK', 0)
    total_organik = petani_data.get('Total_Organik', 0)
    
    # Hitung per ha
    urea_per_ha = total_urea / luas_ha if luas_ha > 0 else 0
    npk_per_ha = total_npk / luas_ha if luas_ha > 0 else 0
    organik_per_ha = total_organik / luas_ha if luas_ha > 0 else 0
    
    comparisons = []
    
    for pupuk, per_ha_value in [('Urea', urea_per_ha), ('NPK', npk_per_ha), ('Organik', organik_per_ha)]:
        if pupuk not in standard or not isinstance(standard[pupuk], dict):
            continue
        
        pupuk_std = standard[pupuk]
        std_max = pupuk_std.get('max', 0)
        
        if per_ha_value > std_max:
            overuse_amount = per_ha_value - std_max
            overuse_pct = (overuse_amount / std_max * 100) if std_max > 0 else 0
            
            if overuse_pct <= 10:
                level = 'Overuse Ringan'
            elif overuse_pct <= 30:
                level = 'Overuse Sedang'
            else:
                level = 'Overuse Tinggi'
            status = 'Overuse'
        else:
            usage_pct = (per_ha_value / std_max * 100) if std_max > 0 else 0
            level = 'Normal'
            status = 'Normal'
        
        comparisons.append({
            'jenis': pupuk,
            'standar_max': std_max,
            'petani_per_ha': per_ha_value,
            'selisih_dari_max': per_ha_value - std_max,
            'persen_dari_max': (per_ha_value / std_max * 100) if std_max > 0 else 0,
            'status': status,
            'level': level
        })
    
    return {
        'has_standard': True,
        'komoditas': komoditas,
        'luas_ha': luas_ha,
        'comparisons': comparisons
    }

def generate_explanation_text(comparison_result: Dict) -> List[str]:
    """
    Generate penjelasan dalam bahasa awam tentang overuse
    
    UPDATED: Hanya jelaskan overuse. Underuse handled separately via statistik.
    
    Returns:
        list of string explanations
    """
    if not comparison_result.get('has_standard'):
        return [comparison_result.get('message', 'Tidak ada penjelasan')]
    
    explanations = []
    komoditas = comparison_result.get('komoditas', '')
    luas_ha = comparison_result.get('luas_ha', 0)
    
    explanations.append(f"**Komoditas {komoditas} dengan luas lahan {luas_ha:.2f} ha**")
    explanations.append("")
    explanations.append("**📌 Filosofi Standar**: Standar pupuk adalah **batas atas hak/alokasi maksimal**, BUKAN target ideal. Petani boleh menggunakan lebih rendah tanpa masalah.")
    explanations.append("")
    
    for comp in comparison_result.get('comparisons', []):
        jenis = comp['jenis']
        std_max = comp['standar_max']
        petani = comp['petani_per_ha']
        persen_dari_max = comp['persen_dari_max']
        status = comp['status']
        level = comp['level']
        
        # Header
        if status == 'Overuse':
            emoji = '🔴'
        else:
            emoji = '🟢'
        
        explanations.append(f"{emoji} **{jenis}**: {level}")
        
        explanations.append(f"- Batas Maksimal: {std_max:.0f} kg/ha")
        explanations.append(f"- Petani menggunakan: {petani:.1f} kg/ha ({persen_dari_max:.0f}% dari maksimal)")
        
        if status == 'Overuse':
            overuse_amount = petani - std_max
            explanations.append(f"- ⚠️ **MELEBIHI BATAS**: +{overuse_amount:.1f} kg/ha di atas maksimal")
            if level == 'Overuse Tinggi':
                explanations.append(f"- **Dampak**: Melebihi alokasi hak subsidi. Perlu evaluasi dan penyesuaian.")
            else:
                explanations.append(f"- Sedikit melebihi batas, perlu konfirmasi data.")
        else:
            remaining = std_max - petani
            explanations.append(f"- ✅ Dalam batas (sisa hak: {remaining:.1f} kg/ha)")
        
        explanations.append("")
    
    return explanations

def get_educational_box() -> str:
    """
    Return konten box edukatif tentang standar pupuk
    
    UPDATED: Jelaskan filosofi baru
    """
    return """
    <div class="info-box">
    <strong>📚 Memahami Standar Pupuk:</strong><br>
    <ul style="margin: 10px 0; padding-left: 20px;">
        <li><strong>Standar Pupuk = Batas Atas Hak</strong>: Standar adalah alokasi maksimal (kg/ha) yang BOLEH ditebus petani, 
        BUKAN kewajiban atau target ideal.</li>
        <li><strong>Overuse (Penggunaan Melebihi Batas)</strong>: Menggunakan pupuk melebihi standar maksimal. 
        Ini adalah anomali kebijakan karena melebihi hak alokasi.</li>
        <li><strong>Di Bawah Standar = BOLEH</strong>: Petani boleh membeli pupuk di bawah standar 
        (misal 150 dari 250 kg/ha) tanpa dianggap kesalahan. Ini hak petani untuk menyesuaikan dengan kondisi lahan.</li>
        <li><strong>Underuse Statistik</strong>: Berbeda dari standar, underuse statistik adalah penggunaan yang 
        SANGAT JAUH di bawah mayoritas petani sejenis (bukan pelanggaran kebijakan, tapi pola yang tidak biasa).</li>
        <li>🔍 <em>Catatan</em>: Setiap petani punya kondisi unik. Yang penting: jangan melebihi batas maksimal.</li>
    </ul>
    </div>
    """

def classify_overuse_level(diff_percent: float) -> Tuple[str, str]:
    """
    Klasifikasi tingkat overuse/underuse
    
    Returns:
        (level, color) tuple
    """
    abs_diff = abs(diff_percent)
    
    if abs_diff <= 10:
        return ('Normal', '#4CAF50')
    elif abs_diff <= 30:
        return ('Ringan', '#FF9800')
    else:
        return ('Tinggi', '#F44336')
