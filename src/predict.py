import pandas as pd
import numpy as np
from pathlib import Path
import json

from .feature_engineering import calculate_total_pupuk_per_ha
from .anomaly_detection import load_anomaly_model, predict_anomaly
from .clustering import load_clustering_model, predict_cluster
from .recommendation import generate_recommendations
from .utils import categorize_anomaly_severity, handle_missing_values

class RDKKPredictor:
    """
    Class untuk prediksi dan analisis data petani baru
    UPDATED: Menggunakan konsep total per ha bukan MT individual
    """
    
    def __init__(self, models_dir='models', standards_path=None):
        """
        Initialize predictor dengan load model dan standards
        """
        self.models_dir = Path(models_dir)
        self.anomaly_model = None
        self.anomaly_scaler = None
        self.anomaly_features = None
        self.cluster_model = None
        self.cluster_scaler = None
        self.cluster_features = None
        self.standards_dict = None
        
        # Load models
        self._load_models()
        
        if standards_path is None:
            standards_path = Path('config/admin_standards.json')
        
        if Path(standards_path).exists():
            self._load_standards(standards_path)
    
    def _load_models(self):
        """Load semua model yang dibutuhkan"""
        try:
            # Load anomaly model
            anomaly_path = self.models_dir / 'isolation_forest.pkl'
            if anomaly_path.exists():
                self.anomaly_model, self.anomaly_scaler, self.anomaly_features = load_anomaly_model(anomaly_path)
                print(f"✓ Anomaly model loaded dari {anomaly_path}")
            else:
                print(f"⚠ Anomaly model tidak ditemukan di {anomaly_path}")
            
            # Load clustering model
            cluster_path = self.models_dir / 'clustering_model.pkl'
            if cluster_path.exists():
                self.cluster_model, self.cluster_scaler, self.cluster_features = load_clustering_model(cluster_path)
                print(f"✓ Clustering model loaded dari {cluster_path}")
            else:
                print(f"⚠ Clustering model tidak ditemukan di {cluster_path}")
        
        except Exception as e:
            print(f"✗ Error loading models: {e}")
    
    def _load_standards(self, standards_path):
        """Load standar pupuk dari file JSON"""
        try:
            with open(standards_path, 'r') as f:
                self.standards_dict = json.load(f)
            print(f"✓ Admin standards loaded dari {standards_path}")
        except Exception as e:
            print(f"⚠ Error loading standards: {e}")
    
    def prepare_input_data(self, data_dict):
        """
        Konversi input dictionary ke DataFrame dengan format yang benar
        
        Parameters:
        - data_dict: dictionary dengan keys seperti:
          {
              'ID_Petani': 'P9999',
              'Desa': 'Desa A',
              'Kelompok_Tani': 'Tani Maju',
              'Komoditas': 'Padi',
              'Luas_Tanah_m2': 5000,
              'Urea_MT1': 50, 'NPK_MT1': 30, 'Organik_MT1': 20,
              'Urea_MT2': 45, 'NPK_MT2': 28, 'Organik_MT2': 18,
              'Urea_MT3': 40, 'NPK_MT3': 25, 'Organik_MT3': 15
          }
        """
        df = pd.DataFrame([data_dict])
        
        df['Total_Urea'] = df['Urea_MT1'] + df['Urea_MT2'] + df['Urea_MT3']
        df['Total_NPK'] = df['NPK_MT1'] + df['NPK_MT2'] + df['NPK_MT3']
        df['Total_Organik'] = df['Organik_MT1'] + df['Organik_MT2'] + df['Organik_MT3']
        
        # Konversi ke hektar
        if 'Luas_ha' not in df.columns and 'Luas_Tanah_m2' in df.columns:
            df['Luas_ha'] = df['Luas_Tanah_m2'] / 10000
        
        return df
    
    def calculate_features(self, df):
        """
        Hitung fitur yang dibutuhkan: HANYA total per ha per jenis pupuk
        """
        df_features = df.copy()
        
        df_features = calculate_total_pupuk_per_ha(df_features)
        
        return df_features
    
    def predict_anomaly(self, df_features):
        """
        Prediksi anomali untuk data baru
        """
        if self.anomaly_model is None:
            return None, None
        
        # Pastikan semua fitur ada
        missing_features = [f for f in self.anomaly_features if f not in df_features.columns]
        if missing_features:
            # Tambahkan kolom yang hilang dengan nilai 0
            for feature in missing_features:
                df_features[feature] = 0
        
        # Prediksi
        predictions, scores = predict_anomaly(
            df_features[self.anomaly_features],
            self.anomaly_model,
            self.anomaly_scaler,
            self.anomaly_features
        )
        
        return predictions, scores
    
    def predict_cluster(self, df_features):
        """
        Prediksi cluster untuk data baru
        """
        if self.cluster_model is None:
            return None
        
        # Pastikan semua fitur ada
        missing_features = [f for f in self.cluster_features if f not in df_features.columns]
        if missing_features:
            for feature in missing_features:
                df_features[feature] = 0
        
        # Prediksi
        cluster_labels = predict_cluster(
            df_features[self.cluster_features],
            self.cluster_model,
            self.cluster_scaler,
            self.cluster_features
        )
        
        return cluster_labels
    
    def predict(self, data_dict, reference_df=None):
        """
        Pipeline lengkap prediksi untuk data baru
        
        Parameters:
        - data_dict: dictionary dengan data petani
        - reference_df: DataFrame referensi (tidak dipakai lagi untuk z-score)
        
        Returns:
        - DataFrame dengan hasil prediksi dan rekomendasi
        """
        # 1. Prepare data
        df = self.prepare_input_data(data_dict)
        
        # 2. Calculate features (hanya total per ha)
        df_features = self.calculate_features(df)
        
        # 3. Predict anomaly
        if self.anomaly_model is not None:
            predictions, scores = self.predict_anomaly(df_features)
            df_features['Anomaly_Prediction'] = predictions
            df_features['Anomaly_Label'] = ['Anomali' if p == -1 else 'Normal' for p in predictions]
            df_features['Anomaly_Score'] = scores
            df_features['Anomaly_Severity'] = categorize_anomaly_severity(scores)
        
        # 4. Predict cluster
        if self.cluster_model is not None:
            cluster_labels = self.predict_cluster(df_features)
            df_features['Cluster'] = cluster_labels
            df_features['Cluster_Label'] = [f'Cluster {c}' if c != -1 else 'Noise' for c in cluster_labels]
        
        # 5. Generate recommendations
        df_result = generate_recommendations(df_features)
        
        return df_result
    
    def batch_predict(self, data_list, reference_df=None):
        """
        Prediksi untuk multiple data sekaligus
        
        Parameters:
        - data_list: list of dictionaries
        - reference_df: DataFrame referensi (tidak dipakai lagi)
        
        Returns:
        - DataFrame dengan hasil prediksi semua data
        """
        results = []
        
        for data_dict in data_list:
            result = self.predict(data_dict, reference_df)
            results.append(result)
        
        df_results = pd.concat(results, ignore_index=True)
        
        return df_results

def create_sample_input():
    """
    Buat sample input untuk testing
    """
    sample = {
        'ID_Petani': 'P_NEW_001',
        'Desa': 'Desa Contoh',
        'Kelompok_Tani': 'Tani Sejahtera',
        'Komoditas': 'Padi',
        'Luas_Tanah_m2': 5000,
        'Urea_MT1': 50,
        'NPK_MT1': 30,
        'Organik_MT1': 20,
        'Urea_MT2': 55,
        'NPK_MT2': 32,
        'Organik_MT2': 22,
        'Urea_MT3': 45,
        'NPK_MT3': 28,
        'Organik_MT3': 18
    }
    
    return sample

if __name__ == "__main__":
    # Test prediction
    print("\n" + "="*60)
    print("TEST PREDICTION SYSTEM")
    print("="*60 + "\n")
    
    # Initialize predictor with standards
    predictor = RDKKPredictor(models_dir='models')
    
    # Create sample input
    sample_data = create_sample_input()
    
    print("\nSample input data:")
    for key, value in sample_data.items():
        print(f"  {key}: {value}")
    
    # Predict
    print("\n" + "-"*60)
    print("Predicting...")
    print("-"*60 + "\n")
    
    result = predictor.predict(sample_data)
    
    # Display results
    print("HASIL PREDIKSI:")
    print("="*60)
    
    if 'Anomaly_Label' in result.columns:
        print(f"\nStatus Anomali: {result['Anomaly_Label'].values[0]}")
        print(f"Skor Anomali: {result['Anomaly_Score'].values[0]:.3f}")
        print(f"Tingkat Keparahan: {result['Anomaly_Severity'].values[0]}")
    
    if 'Cluster_Label' in result.columns:
        print(f"\nCluster: {result['Cluster_Label'].values[0]}")
    
    if 'Rekomendasi' in result.columns:
        print(f"\nRekomendasi:")
        print(f"  {result['Rekomendasi'].values[0]}")
    
    if 'Prioritas' in result.columns:
        print(f"\nPrioritas: {result['Prioritas'].values[0]}")
    
    print("\n" + "="*60)
    
    # Save result
    output_path = Path('output/prediksi_sample.csv')
    result.to_csv(output_path, index=False)
    print(f"\n✓ Hasil prediksi tersimpan: {output_path}")
