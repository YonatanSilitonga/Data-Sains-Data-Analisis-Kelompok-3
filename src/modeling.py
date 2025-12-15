"""
Modeling Module
Model prediksi untuk data petani baru
UPDATED: Menggunakan total per ha bukan MT individual
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from .preprocessing import calculate_totals, calculate_per_ha
from .feature_engineering import calculate_total_pupuk_per_ha
from .anomaly_detection import load_anomaly_model, predict_anomaly
from .standards import get_standard_manager

class FertilizerPredictor:
    """
    Class untuk prediksi data baru
    UPDATED: Tidak lagi menggunakan konsep quota/deviation berbasis MT
    """
    
    def __init__(self, anomaly_model_info: dict, clustering_model_info: dict):
        """
        Args:
            anomaly_model_info: Info dari anomaly detection
            clustering_model_info: Info dari clustering (model, scaler, feature_cols)
        """
        self.clustering_model = clustering_model_info.get('model')
        self.clustering_scaler = clustering_model_info.get('scaler')
        self.clustering_features = clustering_model_info.get('feature_cols', [])
        self.cluster_characteristics = clustering_model_info.get('characteristics', {})
        self.anomaly_model = anomaly_model_info.get('model')
        self.anomaly_scaler = anomaly_model_info.get('scaler')
        self.anomaly_features = anomaly_model_info.get('feature_cols', ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha'])
    
    def preprocess_new_data(self, data: dict) -> pd.DataFrame:
        """
        Preprocess data baru dari form input
        
        Args:
            data: Dict dengan keys seperti ID_Petani, Luas_Tanah_m2, Urea_MT1, etc.
        """
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Calculate totals
        df = calculate_totals(df)
        df = calculate_per_ha(df)
        
        df = calculate_total_pupuk_per_ha(df)
        
        return df
    
    def predict_anomaly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prediksi kategori anomali untuk data baru
        UPDATED: Gunakan Isolation Forest dengan fitur total per ha
        """
        df = df.copy()
        
        if self.anomaly_model and all(f in df.columns for f in self.anomaly_features):
            predictions, scores = predict_anomaly(
                df, 
                self.anomaly_model, 
                self.anomaly_scaler, 
                self.anomaly_features
            )
            
            df['Anomaly_Prediction'] = predictions
            df['Anomaly_Score'] = scores
            df['Anomaly_Label'] = df['Anomaly_Prediction'].apply(
                lambda x: 'Anomali' if x == -1 else 'Normal'
            )
        else:
            df['Anomaly_Label'] = 'unknown'
        
        return df
    
    def predict_cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prediksi cluster untuk data baru
        """
        df = df.copy()
        
        # Prepare features
        X_new = []
        for col in self.clustering_features:
            if col in df.columns:
                X_new.append(df[col].values[0])
            else:
                X_new.append(0)
        
        X_new = np.array(X_new).reshape(1, -1)
        
        # Scale and predict
        X_scaled = self.clustering_scaler.transform(X_new)
        cluster_id = self.clustering_model.predict(X_scaled)[0]
        
        df['Cluster_ID'] = cluster_id
        df['Cluster_Label'] = f'Cluster {cluster_id}'
        
        return df
    
    def generate_recommendation(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate rekomendasi berdasarkan hasil prediksi
        UPDATED: Penjelasan berdasarkan total per ha, tidak menyebut MT
        """
        row = df.iloc[0]
        
        recommendation = {
            'category': row.get('Anomaly_Label', 'unknown'),
            'cluster': row.get('Cluster_Label', 'unknown'),
            'message': '',
            'suggestion': ''
        }
        
        if row.get('Anomaly_Label') == 'Anomali':
            urea = row.get('Urea_per_ha', 0)
            npk = row.get('NPK_per_ha', 0)
            organik = row.get('Organik_per_ha', 0)
            
            recommendation['message'] = f'Penggunaan pupuk tidak sesuai pola mayoritas petani serupa.'
            recommendation['suggestion'] = (
                f'Petani ini menggunakan Urea {urea:.1f} kg/ha, NPK {npk:.1f} kg/ha, '
                f'Organik {organik:.1f} kg/ha. Lakukan verifikasi lapangan untuk memastikan '
                f'data akurat dan tidak ada penyalahgunaan pupuk bersubsidi.'
            )
        
        elif row.get('Anomaly_Label') == 'Normal':
            recommendation['message'] = 'Penggunaan pupuk sesuai dengan pola mayoritas petani serupa.'
            recommendation['suggestion'] = 'Pertahankan pola penggunaan. Monitoring berkala tetap diperlukan.'
        
        else:
            recommendation['message'] = 'Data tidak cukup untuk evaluasi.'
            recommendation['suggestion'] = 'Pastikan data lengkap dan model sudah terlatih.'
        
        # Tambahan info dari cluster
        cluster_id = row.get('Cluster_ID')
        if cluster_id is not None and str(cluster_id) in self.cluster_characteristics:
            recommendation['cluster_info'] = self.cluster_characteristics[str(cluster_id)]
        else:
            recommendation['cluster_info'] = 'Karakteristik cluster tidak tersedia'
        
        return recommendation
    
    def predict(self, data: dict) -> Dict:
        """
        Main prediction method
        
        Returns:
            Dict dengan hasil lengkap prediksi
        """
        # Preprocess
        df = self.preprocess_new_data(data)
        
        # Predict anomaly
        df = self.predict_anomaly(df)
        
        # Predict cluster
        df = self.predict_cluster(df)
        
        # Generate recommendation
        recommendation = self.generate_recommendation(df)
        
        # Compile results
        result = {
            'data': df.to_dict('records')[0],
            'anomaly_category': df['Anomaly_Label'].values[0],
            'cluster_id': int(df['Cluster_ID'].values[0]) if 'Cluster_ID' in df.columns else -1,
            'cluster_label': df['Cluster_Label'].values[0] if 'Cluster_Label' in df.columns else 'Unknown',
            'recommendation': recommendation
        }
        
        return result
