import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Add src to path if needed
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.utils import handle_missing_values
    from src.utils import categorize_anomaly_severity, print_section_header, print_success
except:
    # Fallback for direct imports
    from utils import handle_missing_values
    from utils import categorize_anomaly_severity, print_section_header, print_success

def prepare_features_for_anomaly(df, include_luas=False):
    """
    Persiapkan fitur untuk deteksi anomali - NEW APPROACH
    
    HANYA menggunakan:
    - Urea_per_ha (TOTAL Urea per hektar)
    - NPK_per_ha (TOTAL NPK per hektar)
    - Organik_per_ha (TOTAL Organik per hektar)
    - Luas_ha (opsional)
    
    ❌ TIDAK menggunakan MT, proporsi MT, delta MT, atau z-score
    """
    df_copy = df.copy()
    
    feature_cols = ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha']
    
    # Opsional: tambahkan Luas_ha
    if include_luas:
        luas_col = 'Luas_Tanah_ha' if 'Luas_Tanah_ha' in df_copy.columns else 'Luas_ha'
        if luas_col in df_copy.columns:
            feature_cols.append(luas_col)
    
    # Filter hanya kolom yang ada
    feature_cols = [col for col in feature_cols if col in df_copy.columns]
    
    if len(feature_cols) == 0:
        raise ValueError("Tidak ada fitur yang tersedia. Pastikan feature engineering sudah dijalankan.")
    
    # Extract fitur
    X = df_copy[feature_cols].copy()
    
    # Handle missing values
    X = handle_missing_values(X, strategy='median')
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"→ Menggunakan {len(feature_cols)} fitur: {', '.join(feature_cols)}")
    print(f"→ Basis: TOTAL pupuk per jenis per ha (BUKAN MT)")
    
    return X, feature_cols

def train_isolation_forest(X, contamination=0.1, random_state=42):
    """
    Train Isolation Forest model
    
    Parameters:
    - contamination: proporsi outlier yang diharapkan (0.1 = 10%)
    - random_state: seed untuk hasil yang konsisten dan deterministik
    """
    print_section_header("ISOLATION FOREST - NEW APPROACH")
    
    # Standardisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model dengan random_state yang tetap
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,  # Pastikan hasil konsisten
        n_estimators=100
    )
    
    model.fit(X_scaled)
    
    # Prediksi (-1 = anomali, 1 = normal)
    predictions = model.predict(X_scaled)
    
    # Skor anomali (semakin negatif = semakin anomali)
    scores = model.score_samples(X_scaled)
    
    # Konversi skor ke 0-1 (1 = anomali)
    scores_normalized = 1 - (scores - scores.min()) / (scores.max() - scores.min())
    
    print_success(f"Model trained dengan {len(X)} samples")
    print_success(f"Anomali terdeteksi: {(predictions == -1).sum()} ({(predictions == -1).sum()/len(X)*100:.1f}%)")
    
    return model, scaler, predictions, scores_normalized

def train_local_outlier_factor(X, contamination=0.1, random_state=42):
    """
    Train Local Outlier Factor model
    
    Parameters:
    - contamination: proporsi outlier yang diharapkan (0.1 = 10%)
    - random_state: seed untuk hasil yang konsisten (tidak digunakan LOF, tapi disimpan untuk konsistensi API)
    """
    print_section_header("LOCAL OUTLIER FACTOR - NEW APPROACH")
    
    # Standardisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LocalOutlierFactor(
        contamination=contamination,
        novelty=True,  # Untuk prediksi data baru
        n_neighbors=20
    )
    
    model.fit(X_scaled)
    
    # Prediksi (-1 = anomali, 1 = normal)
    predictions = model.predict(X_scaled)
    
    # Skor anomali
    scores = model.decision_function(X_scaled)
    
    # Konversi skor ke 0-1 (1 = anomali)
    scores_normalized = 1 - (scores - scores.min()) / (scores.max() - scores.min())
    
    print_success(f"Model trained dengan {len(X)} samples")
    print_success(f"Anomali terdeteksi: {(predictions == -1).sum()} ({(predictions == -1).sum()/len(X)*100:.1f}%)")
    
    return model, scaler, predictions, scores_normalized


def detect_anomalies(df, method='isolation_forest', contamination=0.1, include_luas=False, random_state=42):
    """
    Pipeline lengkap deteksi anomali BARU
    
    Hanya menggunakan total pupuk per jenis per ha sebagai basis.
    
    ❌ TIDAK menggunakan MT sebagai penyebab anomali
    ✅ Fokus: Total Urea, NPK, Organik per hektar
    
    Parameters:
    - method: 'isolation_forest' atau 'lof' (Local Outlier Factor)
    - contamination: proporsi anomali yang diharapkan
    - include_luas: apakah Luas_ha dimasukkan sebagai fitur (default: False)
    - random_state: seed untuk hasil deterministik dan konsisten (default: 42)
    """
    print_section_header("ANOMALY DETECTION - NEW APPROACH")
    print("Basis: Total Pupuk per Jenis per Hektar")
    print("❌ TIDAK menggunakan MT, proporsi, atau delta")
    print("="*60)
    
    X, feature_cols_used = prepare_features_for_anomaly(df, include_luas)
    
    print(f"→ Method: {method.upper()}")
    print(f"→ Random State: {random_state} (untuk hasil deterministik)")
    
    # Train model sesuai method dengan random_state
    if method == 'isolation_forest':
        model, scaler, predictions, scores = train_isolation_forest(X, contamination, random_state)
    elif method == 'lof':
        model, scaler, predictions, scores = train_local_outlier_factor(X, contamination, random_state)
    else:
        raise ValueError("Method harus 'isolation_forest' atau 'lof'")
    
    # Tambahkan hasil ke dataframe
    df_result = df.copy()
    df_result['Anomaly_Prediction'] = predictions
    df_result['Anomaly_Label'] = ['Anomali' if p == -1 else 'Normal' for p in predictions]
    df_result['Anomaly_Score'] = scores
    df_result['Anomaly_Severity'] = categorize_anomaly_severity(scores)
    
    print("\n" + "="*60)
    print("RINGKASAN HASIL")
    print("="*60)
    print(f"Total petani: {len(df_result)}")
    print(f"Normal: {(df_result['Anomaly_Label'] == 'Normal').sum()} ({(df_result['Anomaly_Label'] == 'Normal').sum()/len(df_result)*100:.1f}%)")
    print(f"Anomali: {(df_result['Anomaly_Label'] == 'Anomali').sum()} ({(df_result['Anomaly_Label'] == 'Anomali').sum()/len(df_result)*100:.1f}%)")
    print("\nSeverity breakdown:")
    print(df_result['Anomaly_Severity'].value_counts())
    print("="*60 + "\n")
    
    return df_result, model, scaler, feature_cols_used

def save_anomaly_model(model, scaler, feature_cols, output_dir, model_name):
    """Simpan model anomali"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simpan model
    model_path = output_dir / f"{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols
        }, f)
    
    print_success(f"Model tersimpan: {model_path}")

def load_anomaly_model(model_path):
    """Load model anomali"""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['model'], data['scaler'], data['feature_cols']

def predict_anomaly(df, model, scaler, feature_cols):
    """
    Prediksi anomali untuk data baru
    """
    # Persiapkan fitur
    X = df[feature_cols].copy()
    X = handle_missing_values(X, strategy='median')
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Standardisasi
    X_scaled = scaler.transform(X)
    
    # Prediksi
    predictions = model.predict(X_scaled)
    
    # Skor
    if hasattr(model, 'score_samples'):  # Isolation Forest
        scores = model.score_samples(X_scaled)
    else:  # LOF
        scores = model.decision_function(X_scaled)
    
    # Normalisasi skor
    scores_normalized = 1 - (scores - scores.min()) / (scores.max() - scores.min())
    
    return predictions, scores_normalized

def analyze_anomaly_patterns(df_anomaly):
    """
    Analisis pola anomali
    """
    anomalies = df_anomaly[df_anomaly['Anomaly_Label'] == 'Anomali'].copy()
    
    if len(anomalies) == 0:
        print("Tidak ada anomali terdeteksi")
        return None
    
    print_section_header("ANALISIS POLA ANOMALI")
    
    # Distribusi per komoditas
    print("→ Distribusi anomali per komoditas:")
    komoditas_counts = anomalies['Komoditas'].value_counts()
    for komoditas, count in komoditas_counts.items():
        total = len(df_anomaly[df_anomaly['Komoditas'] == komoditas])
        percentage = (count / total) * 100
        print(f"  • {komoditas}: {count}/{total} ({percentage:.1f}%)")
    
    # Rata-rata skor anomali per severity
    print("\n→ Rata-rata skor anomali per tingkat keparahan:")
    severity_scores = df_anomaly.groupby('Anomaly_Severity')['Anomaly_Score'].mean()
    for severity, score in severity_scores.items():
        print(f"  • {severity}: {score:.3f}")
    
    return anomalies

if __name__ == "__main__":
    # Test anomaly detection
    print("Testing Anomaly Detection Module - NEW APPROACH...")
    print("-" * 60)
    
    try:
        from src.data_processing import preprocess_pipeline
        from src.feature_engineering import feature_engineering_pipeline
    except:
        print("Error: Tidak bisa import modules")
        print("Pastikan file dijalankan dari root directory project")
        sys.exit(1)
    
    # Load and process data
    input_path = "data/rdkk_dataset.csv"
    cleaned_path = "output/dataset_cleaned.csv"
    
    df_clean = preprocess_pipeline(input_path, cleaned_path)
    
    if df_clean is not None:
        df_features = feature_engineering_pipeline(df_clean)
        
        # Deteksi anomali dengan random_state tetap
        df_anomaly, model, scaler, feature_cols = detect_anomalies(
            df_features,
            method='isolation_forest',
            contamination=0.1,
            include_luas=False,  # Default: hanya 3 fitur utama
            random_state=42  # Hasil selalu sama untuk data yang sama
        )
        
        # Simpan model
        save_anomaly_model(model, scaler, feature_cols, 'models', 'isolation_forest')
        
        # Simpan hasil
        output_path = Path('output/hasil_anomali.csv')
        df_anomaly.to_csv(output_path, index=False)
        print_success(f"Hasil anomali tersimpan: {output_path}")
        
        # Analisis pola
        analyze_anomaly_patterns(df_anomaly)
        
        print("\n✅ Test anomaly detection berhasil!")
    else:
        print("✗ Preprocessing gagal")
