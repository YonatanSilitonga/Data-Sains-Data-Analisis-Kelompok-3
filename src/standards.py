"""
Standards Manager
Mengelola standar pupuk dinamis dari config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, Optional

class StandardsManager:
    """
    Mengelola standar pupuk per komoditas
    Standar bisa di-update via UI dan tersimpan ke config.yaml
    """
    
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.standards = {}
        self.load_standards()
    
    def load_standards(self):
        """Load standar dari config.yaml"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self.standards = config.get('standar_pupuk', {})
            return True
        except Exception as e:
            print(f"Error loading standards: {e}")
            return False
    
    def save_standards(self):
        """Simpan standar kembali ke config.yaml"""
        try:
            # Load config
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Update standar_pupuk
            config['standar_pupuk'] = self.standards
            
            # Save
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            return True
        except Exception as e:
            print(f"Error saving standards: {e}")
            return False
    
    def get_standard(self, komoditas: str) -> Optional[Dict]:
        """Ambil standar untuk komoditas tertentu"""
        return self.standards.get(komoditas.upper())
    
    def get_all_standards(self) -> Dict:
        """
        Ambil semua standar (kecuali key 'enabled')
        Returns dict dengan hanya komoditas standards, tanpa metadata
        """
        result = {}
        for k, v in self.standards.items():
            if k == 'enabled':
                continue
            if not isinstance(v, dict):
                continue
            # Cek apakah dict ini berisi struktur standar pupuk
            if 'Urea' in v and isinstance(v['Urea'], dict):
                result[k] = v
        return result
    
    def set_standard(self, komoditas: str, urea_min: float, urea_max: float,
                    npk_min: float, npk_max: float, organik_min: float, organik_max: float):
        """Set standar untuk komoditas"""
        komoditas = komoditas.upper()
        self.standards[komoditas] = {
            'Urea': {'min': float(urea_min), 'max': float(urea_max)},
            'NPK': {'min': float(npk_min), 'max': float(npk_max)},
            'Organik': {'min': float(organik_min), 'max': float(organik_max)}
        }
    
    def delete_standard(self, komoditas: str):
        """Hapus standar komoditas"""
        komoditas = komoditas.upper()
        if komoditas in self.standards:
            del self.standards[komoditas]
            return True
        return False
    
    def calculate_status(self, komoditas: str, luas_m2: float, 
                        total_urea: float, total_npk: float, total_organik: float) -> Dict:
        """
        Hitung status over/under/normal use untuk setiap jenis pupuk
        
        Returns:
            dict dengan status_urea, status_npk, status_organik, dll
            ALWAYS returns all required keys even if komoditas not found
        """
        default_result = {
            'status_urea': 'Unknown',
            'status_npk': 'Unknown',
            'status_organik': 'Unknown',
            'jatah_urea_min': 0.0,
            'jatah_urea_max': 0.0,
            'jatah_npk_min': 0.0,
            'jatah_npk_max': 0.0,
            'jatah_organik_min': 0.0,
            'jatah_organik_max': 0.0,
            'luas_ha': luas_m2 / 10000
        }
        
        standard = self.get_standard(komoditas)
        if not standard or not isinstance(standard, dict):
            return default_result
        
        # Konversi ke ha
        luas_ha = luas_m2 / 10000
        
        try:
            urea = standard.get('Urea', {})
            npk = standard.get('NPK', {})
            organik = standard.get('Organik', {})
            
            if not all(isinstance(x, dict) for x in [urea, npk, organik]):
                return default_result
            
            # Hitung jatah (pakai nilai tengah antara min-max)
            jatah_urea_min = urea.get('min', 0) * luas_ha
            jatah_urea_max = urea.get('max', 0) * luas_ha
            jatah_npk_min = npk.get('min', 0) * luas_ha
            jatah_npk_max = npk.get('max', 0) * luas_ha
            jatah_organik_min = organik.get('min', 0) * luas_ha
            jatah_organik_max = organik.get('max', 0) * luas_ha
        except (TypeError, AttributeError):
            return default_result
        
        # Status
        def get_status(actual, min_val, max_val):
            if actual < min_val:
                return 'Underuse'
            elif actual > max_val:
                return 'Overuse'
            else:
                return 'Normal'
        
        return {
            'status_urea': get_status(total_urea, jatah_urea_min, jatah_urea_max),
            'status_npk': get_status(total_npk, jatah_npk_min, jatah_npk_max),
            'status_organik': get_status(total_organik, jatah_organik_min, jatah_organik_max),
            'jatah_urea_min': jatah_urea_min,
            'jatah_urea_max': jatah_urea_max,
            'jatah_npk_min': jatah_npk_min,
            'jatah_npk_max': jatah_npk_max,
            'jatah_organik_min': jatah_organik_min,
            'jatah_organik_max': jatah_organik_max,
            'luas_ha': luas_ha
        }

# Singleton instance
_instance = None

def get_standard_manager(config_path='config.yaml') -> StandardsManager:
    """Get singleton instance of StandardsManager"""
    global _instance
    if _instance is None:
        _instance = StandardsManager(config_path)
    return _instance
