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
    
    # Urea
    if 'Urea' in standard and isinstance(standard['Urea'], dict):
        urea_std = standard['Urea']
        urea_min = urea_std.get('min', 0)
        urea_max = urea_std.get('max', 0)
        urea_mid = (urea_min + urea_max) / 2
        
        urea_diff = urea_per_ha - urea_mid
        urea_diff_pct = (urea_diff / urea_mid * 100) if urea_mid > 0 else 0
        
        # Klasifikasi tingkat overuse/underuse
        if urea_per_ha < urea_min:
            if abs(urea_diff_pct) <= 10:
                level = 'Normal (sedikit di bawah)'
            elif abs(urea_diff_pct) <= 30:
                level = 'Underuse Ringan'
            else:
                level = 'Underuse Tinggi'
            status = 'Underuse'
        elif urea_per_ha > urea_max:
            if abs(urea_diff_pct) <= 10:
                level = 'Normal (sedikit di atas)'
            elif abs(urea_diff_pct) <= 30:
                level = 'Overuse Ringan'
            else:
                level = 'Overuse Tinggi'
            status = 'Overuse'
        else:
            level = 'Normal'
            status = 'Normal'
        
        comparisons.append({
            'jenis': 'Urea',
            'standar_min': urea_min,
            'standar_max': urea_max,
            'standar_mid': urea_mid,
            'petani_per_ha': urea_per_ha,
            'selisih_kg': urea_diff,
            'selisih_persen': urea_diff_pct,
            'status': status,
            'level': level
        })
    
    # NPK
    if 'NPK' in standard and isinstance(standard['NPK'], dict):
        npk_std = standard['NPK']
        npk_min = npk_std.get('min', 0)
        npk_max = npk_std.get('max', 0)
        npk_mid = (npk_min + npk_max) / 2
        
        npk_diff = npk_per_ha - npk_mid
        npk_diff_pct = (npk_diff / npk_mid * 100) if npk_mid > 0 else 0
        
        if npk_per_ha < npk_min:
            if abs(npk_diff_pct) <= 10:
                level = 'Normal (sedikit di bawah)'
            elif abs(npk_diff_pct) <= 30:
                level = 'Underuse Ringan'
            else:
                level = 'Underuse Tinggi'
            status = 'Underuse'
        elif npk_per_ha > npk_max:
            if abs(npk_diff_pct) <= 10:
                level = 'Normal (sedikit di atas)'
            elif abs(npk_diff_pct) <= 30:
                level = 'Overuse Ringan'
            else:
                level = 'Overuse Tinggi'
            status = 'Overuse'
        else:
            level = 'Normal'
            status = 'Normal'
        
        comparisons.append({
            'jenis': 'NPK',
            'standar_min': npk_min,
            'standar_max': npk_max,
            'standar_mid': npk_mid,
            'petani_per_ha': npk_per_ha,
            'selisih_kg': npk_diff,
            'selisih_persen': npk_diff_pct,
            'status': status,
            'level': level
        })
    
    # Organik
    if 'Organik' in standard and isinstance(standard['Organik'], dict):
        organik_std = standard['Organik']
        organik_min = organik_std.get('min', 0)
        organik_max = organik_std.get('max', 0)
        organik_mid = (organik_min + organik_max) / 2
        
        organik_diff = organik_per_ha - organik_mid
        organik_diff_pct = (organik_diff / organik_mid * 100) if organik_mid > 0 else 0
        
        if organik_per_ha < organik_min:
            if abs(organik_diff_pct) <= 10:
                level = 'Normal (sedikit di bawah)'
            elif abs(organik_diff_pct) <= 30:
                level = 'Underuse Ringan'
            else:
                level = 'Underuse Tinggi'
            status = 'Underuse'
        elif organik_per_ha > organik_max:
            if abs(organik_diff_pct) <= 10:
                level = 'Normal (sedikit di atas)'
            elif abs(organik_diff_pct) <= 30:
                level = 'Overuse Ringan'
            else:
                level = 'Overuse Tinggi'
            status = 'Overuse'
        else:
            level = 'Normal'
            status = 'Normal'
        
        comparisons.append({
            'jenis': 'Organik',
            'standar_min': organik_min,
            'standar_max': organik_max,
            'standar_mid': organik_mid,
            'petani_per_ha': organik_per_ha,
            'selisih_kg': organik_diff,
            'selisih_persen': organik_diff_pct,
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
    Generate penjelasan dalam bahasa awam tentang overuse/underuse
    
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
    
    for comp in comparison_result.get('comparisons', []):
        jenis = comp['jenis']
        std_min = comp['standar_min']
        std_max = comp['standar_max']
        std_mid = comp['standar_mid']
        petani = comp['petani_per_ha']
        selisih_kg = comp['selisih_kg']
        selisih_pct = comp['selisih_persen']
        status = comp['status']
        level = comp['level']
        
        # Header
        if status == 'Overuse':
            emoji = '🔴'
        elif status == 'Underuse':
            emoji = '🟡'
        else:
            emoji = '🟢'
        
        explanations.append(f"{emoji} **{jenis}**: {level}")
        
        # Detail
        explanations.append(f"- Standar: {std_min:.0f}-{std_max:.0f} kg/ha (rata-rata {std_mid:.0f} kg/ha)")
        explanations.append(f"- Petani menggunakan: {petani:.1f} kg/ha")
        
        if status == 'Overuse':
            explanations.append(f"- Lebih {abs(selisih_kg):.1f} kg/ha ({abs(selisih_pct):.1f}%) dari standar")
            if level == 'Overuse Tinggi':
                explanations.append(f"- ⚠️ **Dampak**: Penggunaan berlebih dapat menyebabkan pemborosan biaya, pencemaran tanah, dan menurunkan kualitas hasil panen")
            else:
                explanations.append(f"- ℹ️ Sedikit melebihi standar, perlu evaluasi apakah sesuai kondisi lahan")
        elif status == 'Underuse':
            explanations.append(f"- Kurang {abs(selisih_kg):.1f} kg/ha ({abs(selisih_pct):.1f}%) dari standar")
            if level == 'Underuse Tinggi':
                explanations.append(f"- ⚠️ **Dampak**: Kekurangan pupuk dapat menurunkan produktivitas dan hasil panen tidak optimal")
            else:
                explanations.append(f"- ℹ️ Sedikit di bawah standar, mungkin sudah cukup tergantung kondisi lahan")
        else:
            explanations.append(f"- ✅ Penggunaan sesuai standar")
        
        explanations.append("")
    
    return explanations

def get_educational_box() -> str:
    """
    Return konten box edukatif tentang overuse/underuse
    """
    return """
    <div class="info-box">
    <strong>📚 Memahami Overuse & Underuse:</strong><br>
    <ul style="margin: 10px 0; padding-left: 20px;">
        <li><strong>Overuse (Penggunaan Berlebih)</strong>: Menggunakan pupuk melebihi standar rekomendasi. 
        Bisa menyebabkan pemborosan biaya, pencemaran lingkungan, dan menurunkan kualitas hasil panen.</li>
        <li><strong>Underuse (Penggunaan Kurang)</strong>: Menggunakan pupuk di bawah standar rekomendasi. 
        Dapat menurunkan produktivitas karena tanaman kekurangan nutrisi.</li>
        <li><strong>Normal</strong>: Penggunaan sesuai dengan rentang standar yang direkomendasikan.</li>
        <li>🔍 <em>Catatan</em>: Standar adalah pedoman umum. Kebutuhan aktual bisa berbeda tergantung kondisi tanah, 
        iklim, dan praktik pertanian masing-masing petani.</li>
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
    
    # Urea
    if 'Urea' in standard and isinstance(standard['Urea'], dict):
        urea_std = standard['Urea']
        urea_min = urea_std.get('min', 0)
        urea_max = urea_std.get('max', 0)
        urea_mid = (urea_min + urea_max) / 2
        
        urea_diff = urea_per_ha - urea_mid
        urea_diff_pct = (urea_diff / urea_mid * 100) if urea_mid > 0 else 0
        
        # Klasifikasi tingkat overuse/underuse
        if urea_per_ha < urea_min:
            if abs(urea_diff_pct) <= 10:
                level = 'Normal (sedikit di bawah)'
            elif abs(urea_diff_pct) <= 30:
                level = 'Underuse Ringan'
            else:
                level = 'Underuse Tinggi'
            status = 'Underuse'
        elif urea_per_ha > urea_max:
            if abs(urea_diff_pct) <= 10:
                level = 'Normal (sedikit di atas)'
            elif abs(urea_diff_pct) <= 30:
                level = 'Overuse Ringan'
            else:
                level = 'Overuse Tinggi'
            status = 'Overuse'
        else:
            level = 'Normal'
            status = 'Normal'
        
        comparisons.append({
            'jenis': 'Urea',
            'standar_min': urea_min,
            'standar_max': urea_max,
            'standar_mid': urea_mid,
            'petani_per_ha': urea_per_ha,
            'selisih_kg': urea_diff,
            'selisih_persen': urea_diff_pct,
            'status': status,
            'level': level
        })
    
    # NPK
    if 'NPK' in standard and isinstance(standard['NPK'], dict):
        npk_std = standard['NPK']
        npk_min = npk_std.get('min', 0)
        npk_max = npk_std.get('max', 0)
        npk_mid = (npk_min + npk_max) / 2
        
        npk_diff = npk_per_ha - npk_mid
        npk_diff_pct = (npk_diff / npk_mid * 100) if npk_mid > 0 else 0
        
        if npk_per_ha < npk_min:
            if abs(npk_diff_pct) <= 10:
                level = 'Normal (sedikit di bawah)'
            elif abs(npk_diff_pct) <= 30:
                level = 'Underuse Ringan'
            else:
                level = 'Underuse Tinggi'
            status = 'Underuse'
        elif npk_per_ha > npk_max:
            if abs(npk_diff_pct) <= 10:
                level = 'Normal (sedikit di atas)'
            elif abs(npk_diff_pct) <= 30:
                level = 'Overuse Ringan'
            else:
                level = 'Overuse Tinggi'
            status = 'Overuse'
        else:
            level = 'Normal'
            status = 'Normal'
        
        comparisons.append({
            'jenis': 'NPK',
            'standar_min': npk_min,
            'standar_max': npk_max,
            'standar_mid': npk_mid,
            'petani_per_ha': npk_per_ha,
            'selisih_kg': npk_diff,
            'selisih_persen': npk_diff_pct,
            'status': status,
            'level': level
        })
    
    # Organik
    if 'Organik' in standard and isinstance(standard['Organik'], dict):
        organik_std = standard['Organik']
        organik_min = organik_std.get('min', 0)
        organik_max = organik_std.get('max', 0)
        organik_mid = (organik_min + organik_max) / 2
        
        organik_diff = organik_per_ha - organik_mid
        organik_diff_pct = (organik_diff / organik_mid * 100) if organik_mid > 0 else 0
        
        if organik_per_ha < organik_min:
            if abs(organik_diff_pct) <= 10:
                level = 'Normal (sedikit di bawah)'
            elif abs(organik_diff_pct) <= 30:
                level = 'Underuse Ringan'
            else:
                level = 'Underuse Tinggi'
            status = 'Underuse'
        elif organik_per_ha > organik_max:
            if abs(organik_diff_pct) <= 10:
                level = 'Normal (sedikit di atas)'
            elif abs(organik_diff_pct) <= 30:
                level = 'Overuse Ringan'
            else:
                level = 'Overuse Tinggi'
            status = 'Overuse'
        else:
            level = 'Normal'
            status = 'Normal'
        
        comparisons.append({
            'jenis': 'Organik',
            'standar_min': organik_min,
            'standar_max': organik_max,
            'standar_mid': organik_mid,
            'petani_per_ha': organik_per_ha,
            'selisih_kg': organik_diff,
            'selisih_persen': organik_diff_pct,
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
    Generate penjelasan dalam bahasa awam tentang overuse/underuse
    
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
    
    for comp in comparison_result.get('comparisons', []):
        jenis = comp['jenis']
        std_min = comp['standar_min']
        std_max = comp['standar_max']
        std_mid = comp['standar_mid']
        petani = comp['petani_per_ha']
        selisih_kg = comp['selisih_kg']
        selisih_pct = comp['selisih_persen']
        status = comp['status']
        level = comp['level']
        
        # Header
        if status == 'Overuse':
            emoji = '🔴'
        elif status == 'Underuse':
            emoji = '🟡'
        else:
            emoji = '🟢'
        
        explanations.append(f"{emoji} **{jenis}**: {level}")
        
        # Detail
        explanations.append(f"- Standar: {std_min:.0f}-{std_max:.0f} kg/ha (rata-rata {std_mid:.0f} kg/ha)")
        explanations.append(f"- Petani menggunakan: {petani:.1f} kg/ha")
        
        if status == 'Overuse':
            explanations.append(f"- Lebih {abs(selisih_kg):.1f} kg/ha ({abs(selisih_pct):.1f}%) dari standar")
            if level == 'Overuse Tinggi':
                explanations.append(f"- ⚠️ **Dampak**: Penggunaan berlebih dapat menyebabkan pemborosan biaya, pencemaran tanah, dan menurunkan kualitas hasil panen")
            else:
                explanations.append(f"- ℹ️ Sedikit melebihi standar, perlu evaluasi apakah sesuai kondisi lahan")
        elif status == 'Underuse':
            explanations.append(f"- Kurang {abs(selisih_kg):.1f} kg/ha ({abs(selisih_pct):.1f}%) dari standar")
            if level == 'Underuse Tinggi':
                explanations.append(f"- ⚠️ **Dampak**: Kekurangan pupuk dapat menurunkan produktivitas dan hasil panen tidak optimal")
            else:
                explanations.append(f"- ℹ️ Sedikit di bawah standar, mungkin sudah cukup tergantung kondisi lahan")
        else:
            explanations.append(f"- ✅ Penggunaan sesuai standar")
        
        explanations.append("")
    
    return explanations

def get_educational_box() -> str:
    """
    Return konten box edukatif tentang overuse/underuse
    """
    return """
    <div class="info-box">
    <strong>📚 Memahami Overuse & Underuse:</strong><br>
    <ul style="margin: 10px 0; padding-left: 20px;">
        <li><strong>Overuse (Penggunaan Berlebih)</strong>: Menggunakan pupuk melebihi standar rekomendasi. 
        Bisa menyebabkan pemborosan biaya, pencemaran lingkungan, dan menurunkan kualitas hasil panen.</li>
        <li><strong>Underuse (Penggunaan Kurang)</strong>: Menggunakan pupuk di bawah standar rekomendasi. 
        Dapat menurunkan produktivitas karena tanaman kekurangan nutrisi.</li>
        <li><strong>Normal</strong>: Penggunaan sesuai dengan rentang standar yang direkomendasikan.</li>
        <li>🔍 <em>Catatan</em>: Standar adalah pedoman umum. Kebutuhan aktual bisa berbeda tergantung kondisi tanah, 
        iklim, dan praktik pertanian masing-masing petani.</li>
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
