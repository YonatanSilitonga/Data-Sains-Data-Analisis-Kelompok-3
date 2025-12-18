# """
# Model Evaluation Script - RDKK System
# Evaluasi performa model Anomaly Detection dan Clustering
# """

# import pandas as pd
# import numpy as np
# import joblib
# import json
# from pathlib import Path
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import (
#     silhouette_score, 
#     calinski_harabasz_score,
#     davies_bouldin_score,
#     confusion_matrix,
#     classification_report
# )
# from sklearn.decomposition import PCA
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots

# def print_header(text):
#     """Print formatted header"""
#     print("\n" + "="*70)
#     print(f"  {text}")
#     print("="*70)

# def load_data_and_models():
#     """Load processed data and trained models"""
#     print_header("LOADING DATA & MODELS")
    
#     # Load data
#     data_path = Path("output/dataset_final.csv")
#     if not data_path.exists():
#         raise FileNotFoundError(f"Data not found: {data_path}")
    
#     df = pd.read_csv(data_path)
#     print(f"✓ Data loaded: {len(df):,} rows, {len(df.columns)} columns")
    
#     # Load models
#     models_dir = Path("models")
#     models = {}
    
#     # Clustering model
#     scaler_path = models_dir / "scaler.joblib"
#     kmeans_path = models_dir / "kmeans.joblib"
#     features_path = models_dir / "features.json"
    
#     if scaler_path.exists() and kmeans_path.exists() and features_path.exists():
#         models['scaler'] = joblib.load(scaler_path)
#         models['kmeans'] = joblib.load(kmeans_path)
        
#         with open(features_path, 'r') as f:
#             models['features_info'] = json.load(f)
        
#         print(f"✓ Clustering model loaded")
#         print(f"  - Features: {len(models['features_info']['feature_cols'])}")
#         print(f"  - Clusters: {models['features_info']['n_clusters']}")
    
#     # Anomaly model (if exists)
#     anomaly_path = models_dir / "anomaly_isolation_forest_model.joblib"
#     if anomaly_path.exists():
#         anomaly_data = joblib.load(anomaly_path)
#         models['anomaly_model'] = anomaly_data['model']
#         models['anomaly_scaler'] = anomaly_data['scaler']
#         models['anomaly_features'] = anomaly_data['feature_cols']
#         print(f"✓ Anomaly model loaded")
    
#     return df, models

# def evaluate_clustering(df, models):
#     """Evaluate clustering model performance"""
#     print_header("CLUSTERING MODEL EVALUATION")
    
#     if 'Cluster_ID' not in df.columns:
#         print("⚠️  No clustering results found in data")
#         return
    
#     feature_cols = models['features_info']['feature_cols']
#     available_features = [f for f in feature_cols if f in df.columns]
    
#     if len(available_features) < 2:
#         print("⚠️  Insufficient features for evaluation")
#         return
    
#     # Prepare data
#     X = df[available_features].fillna(0)
#     labels = df['Cluster_ID'].values
    
#     print(f"\n📊 Clustering Metrics:")
#     print(f"{'Metric':<30} {'Score':<15} {'Interpretation'}")
#     print("-" * 70)
    
#     # 1. Silhouette Score (-1 to 1, higher is better)
#     silhouette = silhouette_score(X, labels)
#     interpretation = "Excellent" if silhouette > 0.7 else "Good" if silhouette > 0.5 else "Fair" if silhouette > 0.25 else "Poor"
#     print(f"{'Silhouette Score':<30} {silhouette:<15.4f} {interpretation}")
    
#     # 2. Calinski-Harabasz Score (higher is better)
#     calinski = calinski_harabasz_score(X, labels)
#     print(f"{'Calinski-Harabasz Index':<30} {calinski:<15.2f} Higher = Better separation")
    
#     # 3. Davies-Bouldin Score (lower is better)
#     davies = davies_bouldin_score(X, labels)
#     interpretation = "Excellent" if davies < 0.5 else "Good" if davies < 1.0 else "Fair" if davies < 1.5 else "Poor"
#     print(f"{'Davies-Bouldin Index':<30} {davies:<15.4f} {interpretation}")
    
#     # 4. Cluster size distribution
#     print(f"\n📈 Cluster Distribution:")
#     cluster_counts = df['Cluster_ID'].value_counts().sort_index()
    
#     for cluster_id, count in cluster_counts.items():
#         pct = count / len(df) * 100
#         bar = "█" * int(pct / 2)
#         print(f"  Cluster {cluster_id}: {count:>6,} ({pct:>5.1f}%) {bar}")
    
#     # 5. Balance score
#     balance = (cluster_counts.min() / cluster_counts.max()) * 100
#     print(f"\n⚖️  Cluster Balance: {balance:.1f}%")
#     if balance > 50:
#         print("   ✓ Well-balanced clusters")
#     elif balance > 30:
#         print("   ⚠️ Moderately balanced")
#     else:
#         print("   ⚠️ Imbalanced clusters - consider adjusting n_clusters")
    
#     # 6. Inertia (if available)
#     if 'kmeans' in models:
#         inertia = models['kmeans'].inertia_
#         print(f"\n📉 Inertia (within-cluster variance): {inertia:,.2f}")
#         print("   (Lower is better, but consider trade-off with n_clusters)")
    
#     return {
#         'silhouette_score': silhouette,
#         'calinski_harabasz': calinski,
#         'davies_bouldin': davies,
#         'balance_score': balance,
#         'cluster_distribution': cluster_counts.to_dict()
#     }

# def evaluate_anomaly_detection(df, models):
#     """Evaluate anomaly detection model"""
#     print_header("ANOMALY DETECTION EVALUATION")
    
#     if 'Anomaly_Label' not in df.columns:
#         print("⚠️  No anomaly detection results found in data")
#         return
    
#     print(f"\n📊 Anomaly Detection Summary:")
#     print(f"{'Category':<20} {'Count':<12} {'Percentage':<12} {'Visual'}")
#     print("-" * 70)
    
#     # 1. Distribution
#     anomaly_counts = df['Anomaly_Label'].value_counts()
    
#     for label in ['Normal', 'Ringan', 'Sedang', 'Berat']:
#         if label in anomaly_counts.index:
#             count = anomaly_counts[label]
#             pct = count / len(df) * 100
#             bar = "█" * int(pct / 2)
#             print(f"{label:<20} {count:<12,} {pct:<12.1f}% {bar}")
    
#     # 2. Anomaly rate
#     total_anomalies = len(df) - anomaly_counts.get('Normal', 0)
#     anomaly_rate = total_anomalies / len(df) * 100
    
#     print(f"\n🎯 Anomaly Rate: {anomaly_rate:.1f}% ({total_anomalies:,} out of {len(df):,})")
    
#     if anomaly_rate < 5:
#         print("   ℹ️  Very low anomaly rate - model might be too conservative")
#     elif anomaly_rate < 20:
#         print("   ✓ Reasonable anomaly rate")
#     elif anomaly_rate < 40:
#         print("   ⚠️ High anomaly rate - verify if expected")
#     else:
#         print("   ⚠️ Very high anomaly rate - model might be too aggressive")
    
#     # 3. Score distribution
#     if 'Anomaly_Score' in df.columns:
#         print(f"\n📈 Anomaly Score Statistics:")
#         print(f"  Mean:   {df['Anomaly_Score'].mean():.4f}")
#         print(f"  Median: {df['Anomaly_Score'].median():.4f}")
#         print(f"  Std:    {df['Anomaly_Score'].std():.4f}")
#         print(f"  Min:    {df['Anomaly_Score'].min():.4f}")
#         print(f"  Max:    {df['Anomaly_Score'].max():.4f}")
        
#         # Score by category
#         print(f"\n📊 Average Score by Category:")
#         for label in ['Normal', 'Ringan', 'Sedang', 'Berat']:
#             if label in df['Anomaly_Label'].values:
#                 avg_score = df[df['Anomaly_Label'] == label]['Anomaly_Score'].mean()
#                 print(f"  {label:<12}: {avg_score:.4f}")
    
#     return {
#         'anomaly_rate': anomaly_rate,
#         'distribution': anomaly_counts.to_dict(),
#         'total_anomalies': total_anomalies
#     }

# def evaluate_standards_compliance(df):
#     """Evaluate compliance with fertilizer standards (if enabled)"""
#     print_header("STANDARDS COMPLIANCE EVALUATION")
    
#     if 'Final_Status' not in df.columns:
#         print("⚠️  Standards not enabled - skipping compliance evaluation")
#         return
    
#     print(f"\n📊 Status Distribution:")
#     print(f"{'Status':<20} {'Count':<12} {'Percentage':<12} {'Visual'}")
#     print("-" * 70)
    
#     status_counts = df['Final_Status'].value_counts()
    
#     for status in ['Normal', 'Underuse', 'Overuse']:
#         if status in status_counts.index:
#             count = status_counts[status]
#             pct = count / len(df) * 100
#             bar = "█" * int(pct / 2)
            
#             emoji = "✅" if status == 'Normal' else "🟡" if status == 'Underuse' else "🔴"
#             print(f"{emoji} {status:<17} {count:<12,} {pct:<12.1f}% {bar}")
    
#     # Per commodity analysis
#     if 'Komoditas' in df.columns:
#         print(f"\n📈 Status by Commodity:")
        
#         for komoditas in sorted(df['Komoditas'].unique()):
#             df_kom = df[df['Komoditas'] == komoditas]
            
#             if len(df_kom) > 0:
#                 normal = (df_kom['Final_Status'] == 'Normal').sum()
#                 underuse = (df_kom['Final_Status'] == 'Underuse').sum()
#                 overuse = (df_kom['Final_Status'] == 'Overuse').sum()
                
#                 normal_pct = normal / len(df_kom) * 100
#                 underuse_pct = underuse / len(df_kom) * 100
#                 overuse_pct = overuse / len(df_kom) * 100
                
#                 print(f"\n  {komoditas} ({len(df_kom)} petani):")
#                 print(f"    Normal:   {normal:>6} ({normal_pct:>5.1f}%)")
#                 print(f"    Underuse: {underuse:>6} ({underuse_pct:>5.1f}%)")
#                 print(f"    Overuse:  {overuse:>6} ({overuse_pct:>5.1f}%)")
    
#     # Per fertilizer type
#     print(f"\n📊 Status by Fertilizer Type:")
    
#     for pupuk in ['Urea', 'NPK', 'Organik']:
#         status_col = f'Status_{pupuk}'
#         if status_col in df.columns:
#             status_counts = df[status_col].value_counts()
            
#             normal = status_counts.get('Normal', 0)
#             underuse = status_counts.get('Underuse', 0)
#             overuse = status_counts.get('Overuse', 0)
            
#             print(f"\n  {pupuk}:")
#             print(f"    Normal:   {normal:>6,} ({normal/len(df)*100:>5.1f}%)")
#             print(f"    Underuse: {underuse:>6,} ({underuse/len(df)*100:>5.1f}%)")
#             print(f"    Overuse:  {overuse:>6,} ({overuse/len(df)*100:>5.1f}%)")
    
#     return {
#         'status_distribution': status_counts.to_dict()
#     }

# def evaluate_recommendations(df):
#     """Evaluate recommendation system"""
#     print_header("RECOMMENDATION SYSTEM EVALUATION")
    
#     if 'Prioritas' not in df.columns:
#         print("⚠️  No recommendations found in data")
#         return
    
#     print(f"\n📊 Priority Distribution:")
#     print(f"{'Priority':<20} {'Count':<12} {'Percentage':<12} {'Visual'}")
#     print("-" * 70)
    
#     priority_counts = df['Prioritas'].value_counts()
    
#     for priority in ['Tinggi', 'Sedang', 'Rendah']:
#         if priority in priority_counts.index:
#             count = priority_counts[priority]
#             pct = count / len(df) * 100
#             bar = "█" * int(pct / 2)
            
#             emoji = "🔴" if priority == 'Tinggi' else "🟡" if priority == 'Sedang' else "🟢"
#             print(f"{emoji} {priority:<17} {count:<12,} {pct:<12.1f}% {bar}")
    
#     # Coverage analysis
#     print(f"\n📈 Recommendation Coverage:")
#     has_recommendation = df['Rekomendasi'].notna().sum()
#     has_action_plan = df['Action_Plan'].notna().sum() if 'Action_Plan' in df.columns else 0
    
#     print(f"  Has Recommendation: {has_recommendation:>6,} ({has_recommendation/len(df)*100:>5.1f}%)")
#     print(f"  Has Action Plan:    {has_action_plan:>6,} ({has_action_plan/len(df)*100:>5.1f}%)")
    
#     # High priority analysis
#     high_priority = df[df['Prioritas'] == 'Tinggi']
    
#     if len(high_priority) > 0:
#         print(f"\n🚨 High Priority Cases Analysis:")
#         print(f"  Total: {len(high_priority):,} farmers need immediate action")
        
#         if 'Final_Status' in df.columns:
#             high_status = high_priority['Final_Status'].value_counts()
#             print(f"\n  Status breakdown:")
#             for status, count in high_status.items():
#                 print(f"    {status}: {count:,}")
        
#         if 'Anomaly_Label' in df.columns:
#             high_anomaly = high_priority['Anomaly_Label'].value_counts()
#             print(f"\n  Anomaly breakdown:")
#             for label, count in high_anomaly.items():
#                 print(f"    {label}: {count:,}")
    
#     return {
#         'priority_distribution': priority_counts.to_dict(),
#         'coverage_rate': has_recommendation / len(df) * 100
#     }

# def create_evaluation_visualizations(df, models, output_dir='output/evaluation'):
#     """Create evaluation visualizations"""
#     print_header("GENERATING VISUALIZATIONS")
    
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)
    
#     # 1. Clustering visualization with PCA
#     if 'Cluster_ID' in df.columns:
#         print("📊 Creating clustering visualization...")
        
#         feature_cols = models['features_info']['feature_cols']
#         available_features = [f for f in feature_cols if f in df.columns]
        
#         if len(available_features) >= 2:
#             X = df[available_features].fillna(0)
            
#             # PCA
#             pca = PCA(n_components=2)
#             X_pca = pca.fit_transform(X)
            
#             viz_df = pd.DataFrame({
#                 'PC1': X_pca[:, 0],
#                 'PC2': X_pca[:, 1],
#                 'Cluster': df['Cluster_ID'].astype(str),
#                 'Anomaly': df['Anomaly_Label'] if 'Anomaly_Label' in df.columns else 'Unknown'
#             })
            
#             fig = px.scatter(
#                 viz_df,
#                 x='PC1',
#                 y='PC2',
#                 color='Cluster',
#                 symbol='Anomaly',
#                 title=f'Cluster Visualization (PCA) - Variance Explained: {pca.explained_variance_ratio_.sum()*100:.1f}%',
#                 labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
#                        'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'}
#             )
            
#             fig.write_html(output_path / "clustering_pca.html")
#             print(f"  ✓ Saved: {output_path / 'clustering_pca.html'}")
    
#     # 2. Anomaly score distribution
#     if 'Anomaly_Score' in df.columns:
#         print("📊 Creating anomaly distribution...")
        
#         fig = px.histogram(
#             df,
#             x='Anomaly_Score',
#             color='Anomaly_Label',
#             nbins=50,
#             title='Anomaly Score Distribution',
#             labels={'Anomaly_Score': 'Anomaly Score', 'count': 'Frequency'}
#         )
        
#         fig.write_html(output_path / "anomaly_distribution.html")
#         print(f"  ✓ Saved: {output_path / 'anomaly_distribution.html'}")
    
#     # 3. Standards compliance (if enabled)
#     if 'Final_Status' in df.columns and 'Komoditas' in df.columns:
#         print("📊 Creating compliance visualization...")
        
#         compliance_data = df.groupby(['Komoditas', 'Final_Status']).size().reset_index(name='Count')
        
#         fig = px.bar(
#             compliance_data,
#             x='Komoditas',
#             y='Count',
#             color='Final_Status',
#             title='Standards Compliance by Commodity',
#             color_discrete_map={'Normal': '#4caf50', 'Underuse': '#ff9800', 'Overuse': '#f44336'}
#         )
        
#         fig.write_html(output_path / "compliance_by_commodity.html")
#         print(f"  ✓ Saved: {output_path / 'compliance_by_commodity.html'}")
    
#     # 4. Priority distribution
#     if 'Prioritas' in df.columns:
#         print("📊 Creating priority visualization...")
        
#         priority_counts = df['Prioritas'].value_counts()
        
#         fig = px.pie(
#             values=priority_counts.values,
#             names=priority_counts.index,
#             title='Recommendation Priority Distribution',
#             color=priority_counts.index,
#             color_discrete_map={'Tinggi': '#f44336', 'Sedang': '#ff9800', 'Rendah': '#4caf50'}
#         )
        
#         fig.write_html(output_path / "priority_distribution.html")
#         print(f"  ✓ Saved: {output_path / 'priority_distribution.html'}")
    
#     print(f"\n✓ All visualizations saved to: {output_path}")

# def generate_evaluation_report(df, models, results, output_path='output/evaluation/evaluation_report.txt'):
#     """Generate comprehensive evaluation report"""
#     print_header("GENERATING EVALUATION REPORT")
    
#     output_file = Path(output_path)
#     output_file.parent.mkdir(parents=True, exist_ok=True)
    
#     with open(output_file, 'w', encoding='utf-8') as f:
#         f.write("="*70 + "\n")
#         f.write("  RDKK SYSTEM - MODEL EVALUATION REPORT\n")
#         f.write("="*70 + "\n\n")
        
#         f.write(f"Dataset Size: {len(df):,} farmers\n")
#         f.write(f"Number of Features: {len(models['features_info']['feature_cols'])}\n")
#         f.write(f"Number of Clusters: {models['features_info']['n_clusters']}\n\n")
        
#         # Clustering results
#         f.write("-"*70 + "\n")
#         f.write("CLUSTERING PERFORMANCE\n")
#         f.write("-"*70 + "\n")
        
#         if 'clustering' in results:
#             f.write(f"Silhouette Score:        {results['clustering']['silhouette_score']:.4f}\n")
#             f.write(f"Calinski-Harabasz:       {results['clustering']['calinski_harabasz']:.2f}\n")
#             f.write(f"Davies-Bouldin:          {results['clustering']['davies_bouldin']:.4f}\n")
#             f.write(f"Cluster Balance:         {results['clustering']['balance_score']:.1f}%\n\n")
        
#         # Anomaly results
#         f.write("-"*70 + "\n")
#         f.write("ANOMALY DETECTION PERFORMANCE\n")
#         f.write("-"*70 + "\n")
        
#         if 'anomaly' in results:
#             f.write(f"Anomaly Rate:            {results['anomaly']['anomaly_rate']:.1f}%\n")
#             f.write(f"Total Anomalies:         {results['anomaly']['total_anomalies']:,}\n\n")
        
#         # Standards compliance
#         if 'standards' in results:
#             f.write("-"*70 + "\n")
#             f.write("STANDARDS COMPLIANCE\n")
#             f.write("-"*70 + "\n")
            
#             for status, count in results['standards']['status_distribution'].items():
#                 pct = count / len(df) * 100
#                 f.write(f"{status:<15}: {count:>6,} ({pct:>5.1f}%)\n")
#             f.write("\n")
        
#         # Recommendations
#         if 'recommendations' in results:
#             f.write("-"*70 + "\n")
#             f.write("RECOMMENDATION SYSTEM\n")
#             f.write("-"*70 + "\n")
            
#             for priority, count in results['recommendations']['priority_distribution'].items():
#                 pct = count / len(df) * 100
#                 f.write(f"{priority:<15}: {count:>6,} ({pct:>5.1f}%)\n")
            
#             f.write(f"\nCoverage Rate:           {results['recommendations']['coverage_rate']:.1f}%\n")
    
#     print(f"✓ Report saved: {output_file}")

# def main():
#     """Main evaluation function"""
#     try:
#         # Load data and models
#         df, models = load_data_and_models()
        
#         # Run evaluations
#         results = {}
        
#         results['clustering'] = evaluate_clustering(df, models)
#         results['anomaly'] = evaluate_anomaly_detection(df, models)
#         results['standards'] = evaluate_standards_compliance(df)
#         results['recommendations'] = evaluate_recommendations(df)
        
#         # Create visualizations
#         create_evaluation_visualizations(df, models)
        
#         # Generate report
#         generate_evaluation_report(df, models, results)
        
#         print_header("EVALUATION COMPLETE")
#         print("✓ All evaluations finished successfully")
#         print("✓ Check 'output/evaluation/' for detailed reports and visualizations")
        
#     except Exception as e:
#         print(f"\n❌ Error during evaluation: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()

"""
Model Evaluation Script - RDKK System
Visualisasi disimpan dalam format PNG (tanpa HTML)
Unsupervised Evaluation Only
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# ======================================================
# GLOBAL CONFIG
# ======================================================
OUTPUT_DIR = Path("output/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("default")


# ======================================================
# UTIL
# ======================================================
def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ======================================================
# LOAD DATA & MODELS
# ======================================================
def load_all():
    header("LOADING DATA & MODELS")

    df = pd.read_csv("output/dataset_final.csv")
    print(f"✓ Dataset loaded: {df.shape}")

    with open("models/features.json") as f:
        feature_info = json.load(f)

    kmeans = joblib.load("models/kmeans.joblib")

    anomaly_bundle = joblib.load("models/anomaly_isolation_forest.pkl")

    return df, feature_info, kmeans, anomaly_bundle


# ======================================================
# CLUSTERING EVALUATION
# ======================================================
def evaluate_clustering(df, feature_info):
    header("CLUSTERING EVALUATION")

    feature_cols = feature_info["feature_cols"]
    X = df[feature_cols].fillna(0)
    labels = df["Cluster_ID"]

    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    davies = davies_bouldin_score(X, labels)

    print(f"Silhouette Score       : {silhouette:.4f}")
    print(f"Calinski-Harabasz      : {calinski:.2f}")
    print(f"Davies-Bouldin Index   : {davies:.4f}")

    # -----------------------------
    # Cluster Distribution Plot
    # -----------------------------
    cluster_counts = labels.value_counts().sort_index()

    plt.figure(figsize=(7, 4))
    cluster_counts.plot(kind="bar")
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Farmers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_distribution.png", dpi=300)
    plt.close()

    print("✓ Saved: cluster_distribution.png")

    # -----------------------------
    # PCA Scatter Plot
    # -----------------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        s=10
    )
    plt.title("PCA Projection of Clusters")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_pca.png", dpi=300)
    plt.close()

    print("✓ Saved: cluster_pca.png")

    return {
        "silhouette": silhouette,
        "calinski": calinski,
        "davies": davies,
    }


# ======================================================
# ANOMALY DETECTION EVALUATION
# ======================================================
def evaluate_anomaly(df):
    header("ANOMALY DETECTION EVALUATION")

    counts = df["Anomaly_Label"].value_counts()

    # -----------------------------
    # Anomaly Proportion
    # -----------------------------
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Anomaly vs Normal Distribution")
    plt.xlabel("Label")
    plt.ylabel("Number of Farmers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "anomaly_distribution.png", dpi=300)
    plt.close()

    print("✓ Saved: anomaly_distribution.png")

    # -----------------------------
    # Anomaly Score Histogram
    # -----------------------------
    plt.figure(figsize=(6, 4))
    plt.hist(df["Anomaly_Score"], bins=50)
    plt.title("Distribution of Anomaly Scores")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "anomaly_score_histogram.png", dpi=300)
    plt.close()

    print("✓ Saved: anomaly_score_histogram.png")

    anomaly_rate = (df["Anomaly_Label"] != "Normal").mean() * 100
    print(f"Anomaly Rate: {anomaly_rate:.2f}%")

    return {"anomaly_rate": anomaly_rate}


# ======================================================
# STANDARDS EVALUATION
# ======================================================
def evaluate_standards(df):
    header("STANDARDS COMPLIANCE")

    counts = df["Final_Status"].value_counts()

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Standards Compliance Status")
    plt.xlabel("Status")
    plt.ylabel("Number of Farmers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "standards_compliance.png", dpi=300)
    plt.close()

    print("✓ Saved: standards_compliance.png")

    return {"distribution": counts.to_dict()}


# ======================================================
# MAIN
# ======================================================
def main():
    df, feature_info, _, _ = load_all()

    results = {}
    results["clustering"] = evaluate_clustering(df, feature_info)
    results["anomaly"] = evaluate_anomaly(df)
    results["standards"] = evaluate_standards(df)

    header("EVALUATION FINISHED")
    print("✓ All visualizations saved as PNG")
    print("✓ Safe for reports & documentation")


if __name__ == "__main__":
    main()
