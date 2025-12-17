"""
Main Pipeline - RDKK Fertilizer Management System
Complete ML Pipeline: ETL → Feature Engineering → Anomaly Detection → Clustering → Recommendation
"""

import sys
import yaml
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_data, validate_columns, prepare_data
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import feature_engineering_pipeline
from src.anomaly_detection import detect_anomalies, save_anomaly_model
from src.clustering import run_clustering_pipeline
from src.recommendation import generate_recommendations, print_recommendation_summary, export_recommendations_report
from src.standards import get_standard_manager
from src.utils import print_section_header, print_success, print_error

def safe_json(obj):
    """Convert object to JSON-safe format"""
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return safe_json(obj.tolist())
    else:
        return obj

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print_success(f"Config loaded: {config_path}")
        return config
    except Exception as e:
        print_error(f"Error loading config: {e}")
        return None

def run_full_pipeline(config):
    """
    Full ML Pipeline End-to-End:
    1. Load & Validate Data
    2. Preprocessing & Labeling (dengan/tanpa standar)
    3. Feature Engineering
    4. Anomaly Detection (IsolationForest - no unrealistic defaults)
    5. Clustering (KMeans/DBSCAN)
    6. Recommendation Generation
    7. Save Models & Results
    """
    print_section_header("RDKK FULL ML PIPELINE - ANALISIS PUPUK SUBSIDI")
    
    # Setup paths
    input_path = config['data']['input_csv']
    output_dir = Path(config['data']['output_dir'])
    models_dir = Path(config['models']['output_dir'])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # STEP 0: LOAD STANDARDS (OPTIONAL)
    # ==========================================
    print_section_header("STEP 0: LOAD STANDARDS")
    
    standards_enabled = config.get('standar_pupuk', {}).get('enabled', True)
    
    standards_manager = get_standard_manager(config_path='config.yaml')
    
    if standards_enabled:
        all_standards = standards_manager.get_all_standards()
        
        commodity_standards = {}
        for k, v in all_standards.items():
            if k == 'enabled':
                continue
            if not isinstance(v, dict):
                continue
            commodity_standards[k] = v
        
        if commodity_standards:
            print_success(f"Standar pupuk AKTIF - {len(commodity_standards)} komoditas")
            for komoditas, standard in commodity_standards.items():
                if isinstance(standard, dict) and 'Urea' in standard:
                    urea = standard['Urea']
                    if isinstance(urea, dict) and 'min' in urea and 'max' in urea:
                        print(f"  {komoditas}: Urea {urea['min']}-{urea['max']} kg/ha")
        else:
            print_success("Standar pupuk AKTIF - belum ada standar yang didefinisikan")
    else:
        print_success("Standar pupuk NON-AKTIF - analisis berbasis data aktual")
    
    # ==========================================
    # STEP 1: LOAD & VALIDATE DATA
    # ==========================================
    print_section_header("STEP 1: LOAD & VALIDATE DATA")
    
    try:
        df = load_data(input_path)
        print_success(f"Data loaded: {len(df):,} rows")
        
        is_valid, missing = validate_columns(df)
        if not is_valid:
            print_error(f"Missing columns: {missing}")
            return
        
        df = prepare_data(df)
    except Exception as e:
        print_error(f"Error loading data: {e}")
        return
    
    # ==========================================
    # STEP 2: PREPROCESSING & LABELING
    # ==========================================
    print_section_header("STEP 2: PREPROCESSING & STATUS LABELING")
    
    df_clean = preprocess_pipeline(
        df, 
        standards_manager=standards_manager if standards_enabled else None
    )
    print_success(f"Data processed: {len(df_clean):,} rows")
    
    if standards_enabled and 'Final_Status' in df_clean.columns:
        print("\nStatus Summary (berdasarkan standar):")
        status_counts = df_clean['Final_Status'].value_counts()
        for status, count in status_counts.items():
            pct = count / len(df_clean) * 100
            print(f"  {status}: {count:,} ({pct:.1f}%)")
    
    # Save cleaned data
    cleaned_path = output_dir / "dataset_cleaned.csv"
    df_clean.to_csv(cleaned_path, index=False)
    print_success(f"Saved: {cleaned_path}")
    
    # ==========================================
    # STEP 3: FEATURE ENGINEERING
    # ==========================================
    print_section_header("STEP 3: FEATURE ENGINEERING")
    
    standards_dict = {}
    if standards_enabled:
        all_standards = standards_manager.get_all_standards()
        
        # Pass standards directly without calculating midpoints
        for komoditas, std in all_standards.items():
            if not isinstance(std, dict):
                continue
                
            standards_dict[komoditas] = {}
            for pupuk in ['Urea', 'NPK', 'Organik']:
                if pupuk in std and isinstance(std[pupuk], dict):
                    pupuk_std = std[pupuk]
                    if 'min' in pupuk_std and 'max' in pupuk_std:
                        if isinstance(pupuk_std['min'], (int, float)) and isinstance(pupuk_std['max'], (int, float)):
                            # Just pass the dict, let feature engineering decide what to do
                            standards_dict[komoditas][pupuk] = pupuk_std
    
    df_features = feature_engineering_pipeline(
        df_clean, 
        standards_dict=standards_dict if standards_enabled else None
    )
    print_success(f"Feature engineering complete: {len(df_features.columns)} total columns")
    
    # Save features
    features_path = output_dir / "dataset_with_features.csv"
    df_features.to_csv(features_path, index=False)
    print_success(f"Saved: {features_path}")
    
    # ==========================================
    # STEP 4: ANOMALY DETECTION
    # ==========================================
    print_section_header("STEP 4: ANOMALY DETECTION (ML-based)")
    
    anomaly_config = config.get('anomaly', {})
    anomaly_method = anomaly_config.get('method', 'isolation_forest')
    contamination = anomaly_config.get('contamination', 0.15)
    
    print(f"Method: {anomaly_method.upper()}")
    print(f"Expected outliers: {contamination*100:.0f}%")
    print("ℹ️  Anomaly detection berbasis ML - tidak bergantung pada min/max default")
    
    df_anomaly, model, scaler, feature_cols = detect_anomalies(
        df_features,
        method=anomaly_method,
        contamination=contamination,
        random_state=42  # Ensure consistent results
    )
    
    print_success(f"Anomaly detection complete")
    
    # Save anomaly model
    if model is not None and scaler is not None:
        save_anomaly_model(model, scaler, feature_cols, models_dir, f'anomaly_{anomaly_method}')
    
    # Save anomaly results
    anomaly_path = output_dir / "dataset_with_anomaly.csv"
    df_anomaly.to_csv(anomaly_path, index=False)
    print_success(f"Saved: {anomaly_path}")
    
    # ==========================================
    # STEP 5: CLUSTERING
    # ==========================================
    print_section_header("STEP 5: CLUSTERING")
    
    clustering_config = config['clustering']
    df_clustered, model_info, characteristics = run_clustering_pipeline(
        df_anomaly,
        n_clusters=clustering_config.get('n_clusters'),
        random_state=42  # Ensure consistent cluster assignments
    )
    
    print_success(f"Clustering complete: {model_info['n_clusters']} clusters")
    print(f"Silhouette Score: {model_info['silhouette_score']:.3f}")
    
    # Print characteristics
    print("\nCluster Characteristics:")
    for cluster_id, desc in characteristics.items():
        cluster_data = df_clustered[df_clustered['Cluster_ID'] == int(cluster_id)]
        print(f"  {desc} ({len(cluster_data)} petani)")
    
    # ==========================================
    # STEP 6: RECOMMENDATION GENERATION
    # ==========================================
    print_section_header("STEP 6: GENERATE RECOMMENDATIONS")
    
    df_final = generate_recommendations(
        df_clustered, 
        standards_enabled=standards_enabled,
        include_details=True
    )
    
    print_recommendation_summary(df_final)
    
    # Save final dataset
    final_path = output_dir / "dataset_final.csv"
    df_final.to_csv(final_path, index=False)
    print_success(f"Saved: {final_path}")
    
    # Export recommendations report
    recommendations_path = output_dir / "recommendations_report.csv"
    export_recommendations_report(df_final, recommendations_path)
    
    # ==========================================
    # STEP 7: SAVE MODELS
    # ==========================================
    print_section_header("STEP 7: SAVE MODELS")
    
    scaler_path = models_dir / config['models']['scaler_file']
    joblib.dump(model_info['scaler'], scaler_path)
    print_success(f"Saved: {scaler_path}")
    
    kmeans_path = models_dir / config['models']['kmeans_file']
    joblib.dump(model_info['model'], kmeans_path)
    print_success(f"Saved: {kmeans_path}")
    
    features_path = models_dir / config['models']['features_file']
    with open(features_path, 'w') as f:
        json.dump(safe_json({
            'feature_cols': model_info['feature_cols'],
            'n_clusters': model_info['n_clusters'],
            'characteristics': characteristics,
            'standards_enabled': standards_enabled
        }), f, indent=2)
    print_success(f"Saved: {features_path}")
    
    # ==========================================
    # COMPLETE
    # ==========================================
    print_section_header("PIPELINE COMPLETE")
    
    print("📁 Output Files:")
    print(f"  - {cleaned_path}")
    print(f"  - {features_path}")
    print(f"  - {anomaly_path}")
    print(f"  - {final_path}")
    print(f"  - {recommendations_path}")
    print(f"\n🤖 Models:")
    print(f"  - {scaler_path}")
    print(f"  - {kmeans_path}")
    print(f"\n🚀 Next Steps:")
    print("  Run dashboard: streamlit run app.py")
    print("="*60 + "\n")

def main():
    """Main function"""
    # Load config
    config = load_config()
    if config is None:
        sys.exit(1)
    
    # Run full pipeline
    try:
        run_full_pipeline(config)
    except Exception as e:
        print_error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
