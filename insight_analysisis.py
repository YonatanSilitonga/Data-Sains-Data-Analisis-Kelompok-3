"""
RDKK Comprehensive Insight Analysis
====================================
Script untuk menghasilkan insight mendalam dari hasil pipeline ML
meliputi: statistik deskriptif, distribusi anomali, cluster analysis,
compliance assessment, dan rekomendasi strategis.

Usage: python insight_analysis.py
"""

import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RDKKInsightAnalyzer:
    """Analyzer untuk menghasilkan insight dari data RDKK"""
    
    def __init__(self, config_path='config.yaml'):
        self.config = self.load_config(config_path)
        self.output_dir = Path(self.config['data']['output_dir'])
        self.models_dir = Path(self.config['models']['output_dir'])
        self.insights = {}
        
    def load_config(self, config_path):
        """Load configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_data(self):
        """Load final processed data"""
        try:
            final_path = self.output_dir / "dataset_final.csv"
            df = pd.read_csv(final_path)
            print(f"✓ Data loaded: {len(df):,} records")
            return df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def load_model_info(self):
        """Load model metadata"""
        try:
            features_path = self.models_dir / self.config['models']['features_file']
            with open(features_path, 'r') as f:
                model_info = json.load(f)
            print(f"✓ Model info loaded")
            return model_info
        except Exception as e:
            print(f"✗ Error loading model info: {e}")
            return None
    
    def analyze_descriptive_statistics(self, df):
        """Analisis statistik deskriptif"""
        print("\n" + "="*70)
        print("📊 DESCRIPTIVE STATISTICS ANALYSIS")
        print("="*70)
        
        stats = {
            'total_farmers': len(df),
            'total_commodities': df['Komoditas'].nunique() if 'Komoditas' in df.columns else 0,
            'total_villages': df['Desa'].nunique() if 'Desa' in df.columns else 0,
        }
        
        # Land statistics
        if 'Luas_Tanah_ha' in df.columns:
            stats['land_area'] = {
                'total_ha': df['Luas_Tanah_ha'].sum(),
                'mean_ha': df['Luas_Tanah_ha'].mean(),
                'median_ha': df['Luas_Tanah_ha'].median(),
                'min_ha': df['Luas_Tanah_ha'].min(),
                'max_ha': df['Luas_Tanah_ha'].max(),
                'std_ha': df['Luas_Tanah_ha'].std()
            }
        
        # Fertilizer usage statistics
        fertilizer_cols = ['Total_Urea', 'Total_NPK', 'Total_Organik', 'Total_Pupuk']
        for col in fertilizer_cols:
            if col in df.columns:
                stats[col.lower()] = {
                    'total_kg': df[col].sum(),
                    'mean_kg': df[col].mean(),
                    'median_kg': df[col].median(),
                    'std_kg': df[col].std()
                }
        
        # Intensity statistics
        intensity_cols = ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha', 'Total_per_ha']
        for col in intensity_cols:
            if col in df.columns:
                stats[col.lower()] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
        
        self.insights['descriptive_stats'] = stats
        
        # Print summary
        print(f"\n📈 Basic Statistics:")
        print(f"  Total Farmers: {stats['total_farmers']:,}")
        print(f"  Commodities: {stats['total_commodities']}")
        print(f"  Villages: {stats['total_villages']}")
        
        if 'land_area' in stats:
            print(f"\n🌾 Land Statistics:")
            print(f"  Total Area: {stats['land_area']['total_ha']:,.2f} ha")
            print(f"  Average: {stats['land_area']['mean_ha']:.2f} ha")
            print(f"  Median: {stats['land_area']['median_ha']:.2f} ha")
            print(f"  Range: {stats['land_area']['min_ha']:.2f} - {stats['land_area']['max_ha']:.2f} ha")
        
        if 'total_pupuk' in stats:
            print(f"\n💊 Fertilizer Usage:")
            print(f"  Total: {stats['total_pupuk']['total_kg']:,.2f} kg")
            print(f"  Average per Farmer: {stats['total_pupuk']['mean_kg']:.2f} kg")
            print(f"  Median: {stats['total_pupuk']['median_kg']:.2f} kg")
        
        return stats
    
    def analyze_anomaly_distribution(self, df):
        """Analisis distribusi anomali"""
        print("\n" + "="*70)
        print("🔍 ANOMALY DETECTION ANALYSIS")
        print("="*70)
        
        if 'Anomaly_Label' not in df.columns:
            print("⚠️  No anomaly labels found")
            return None
        
        anomaly_stats = {
            'distribution': df['Anomaly_Label'].value_counts().to_dict(),
            'percentages': (df['Anomaly_Label'].value_counts() / len(df) * 100).to_dict()
        }
        
        if 'Anomaly_Score' in df.columns:
            anomaly_stats['scores'] = {
                'mean': df['Anomaly_Score'].mean(),
                'median': df['Anomaly_Score'].median(),
                'std': df['Anomaly_Score'].std(),
                'min': df['Anomaly_Score'].min(),
                'max': df['Anomaly_Score'].max()
            }
            
            # Statistics by category
            for label in df['Anomaly_Label'].unique():
                mask = df['Anomaly_Label'] == label
                anomaly_stats[f'score_{label}'] = {
                    'mean': df.loc[mask, 'Anomaly_Score'].mean(),
                    'median': df.loc[mask, 'Anomaly_Score'].median(),
                    'count': mask.sum()
                }
        
        self.insights['anomaly_analysis'] = anomaly_stats
        
        # Print summary
        print(f"\n📊 Anomaly Distribution:")
        for label, count in anomaly_stats['distribution'].items():
            pct = anomaly_stats['percentages'][label]
            print(f"  {label}: {count:,} ({pct:.1f}%)")
        
        if 'scores' in anomaly_stats:
            print(f"\n📈 Anomaly Scores:")
            print(f"  Mean: {anomaly_stats['scores']['mean']:.4f}")
            print(f"  Median: {anomaly_stats['scores']['median']:.4f}")
            print(f"  Range: {anomaly_stats['scores']['min']:.4f} to {anomaly_stats['scores']['max']:.4f}")
        
        # Interpretation
        total_anomalies = sum([v for k, v in anomaly_stats['distribution'].items() if k != 'Normal'])
        anomaly_rate = total_anomalies / len(df) * 100
        
        print(f"\n💡 Interpretation:")
        if anomaly_rate > 30:
            print(f"  ⚠️  HIGH anomaly rate ({anomaly_rate:.1f}%) - Significant deviations detected")
            print(f"     Recommendation: Conduct thorough field verification")
        elif anomaly_rate > 15:
            print(f"  ⚠️  MODERATE anomaly rate ({anomaly_rate:.1f}%) - Some concerns identified")
            print(f"     Recommendation: Targeted monitoring required")
        else:
            print(f"  ✓  NORMAL anomaly rate ({anomaly_rate:.1f}%) - Most farmers follow standard patterns")
            print(f"     Recommendation: Continue regular monitoring")
        
        return anomaly_stats
    
    def analyze_clustering_results(self, df):
        """Analisis hasil clustering"""
        print("\n" + "="*70)
        print("🎯 CLUSTERING ANALYSIS")
        print("="*70)
        
        if 'Cluster_ID' not in df.columns:
            print("⚠️  No cluster information found")
            return None
        
        cluster_stats = {
            'n_clusters': df['Cluster_ID'].nunique(),
            'distribution': df['Cluster_ID'].value_counts().sort_index().to_dict(),
            'percentages': (df['Cluster_ID'].value_counts() / len(df) * 100).sort_index().to_dict()
        }
        
        # Statistics per cluster
        cluster_stats['profiles'] = {}
        for cluster_id in sorted(df['Cluster_ID'].unique()):
            cluster_data = df[df['Cluster_ID'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100
            }
            
            # Land characteristics
            if 'Luas_Tanah_ha' in cluster_data.columns:
                profile['land_mean'] = cluster_data['Luas_Tanah_ha'].mean()
                profile['land_median'] = cluster_data['Luas_Tanah_ha'].median()
            
            # Fertilizer intensity
            if 'Total_per_ha' in cluster_data.columns:
                profile['intensity_mean'] = cluster_data['Total_per_ha'].mean()
                profile['intensity_median'] = cluster_data['Total_per_ha'].median()
            
            # Individual fertilizer types
            for fert in ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha']:
                if fert in cluster_data.columns:
                    profile[f'{fert}_mean'] = cluster_data[fert].mean()
            
            # Anomaly distribution in cluster
            if 'Anomaly_Label' in cluster_data.columns:
                profile['anomaly_rate'] = (cluster_data['Anomaly_Label'] != 'Normal').mean() * 100
            
            # Status distribution (if standards enabled)
            if 'Final_Status' in cluster_data.columns:
                profile['status_dist'] = cluster_data['Final_Status'].value_counts().to_dict()
            
            cluster_stats['profiles'][int(cluster_id)] = profile
        
        # Calculate balance score
        cluster_sizes = df['Cluster_ID'].value_counts()
        balance_score = (cluster_sizes.min() / cluster_sizes.max()) * 100 if cluster_sizes.max() > 0 else 0
        cluster_stats['balance_score'] = balance_score
        
        self.insights['clustering_analysis'] = cluster_stats
        
        # Print summary
        print(f"\n📊 Cluster Distribution:")
        print(f"  Number of Clusters: {cluster_stats['n_clusters']}")
        print(f"  Balance Score: {balance_score:.1f}%")
        print()
        
        for cluster_id, profile in cluster_stats['profiles'].items():
            print(f"  Cluster {cluster_id}: {profile['size']:,} farmers ({profile['percentage']:.1f}%)")
            if 'land_mean' in profile:
                print(f"    └─ Average Land: {profile['land_mean']:.2f} ha")
            if 'intensity_mean' in profile:
                print(f"    └─ Average Intensity: {profile['intensity_mean']:.1f} kg/ha")
            if 'anomaly_rate' in profile:
                print(f"    └─ Anomaly Rate: {profile['anomaly_rate']:.1f}%")
        
        return cluster_stats
    
    def analyze_standards_compliance(self, df):
        """Analisis kepatuhan terhadap standar"""
        print("\n" + "="*70)
        print("✅ STANDARDS COMPLIANCE ANALYSIS")
        print("="*70)
        
        if 'Final_Status' not in df.columns:
            print("⚠️  Standards compliance not available (standards might be disabled)")
            return None
        
        compliance_stats = {
            'overall': df['Final_Status'].value_counts().to_dict(),
            'percentages': (df['Final_Status'].value_counts() / len(df) * 100).to_dict()
        }
        
        # Per commodity analysis
        if 'Komoditas' in df.columns:
            compliance_stats['by_commodity'] = {}
            for commodity in df['Komoditas'].unique():
                commodity_data = df[df['Komoditas'] == commodity]
                compliance_stats['by_commodity'][commodity] = {
                    'total': len(commodity_data),
                    'distribution': commodity_data['Final_Status'].value_counts().to_dict(),
                    'percentages': (commodity_data['Final_Status'].value_counts() / len(commodity_data) * 100).to_dict()
                }
        
        # Per fertilizer type
        for fert in ['Urea', 'NPK', 'Organik']:
            status_col = f'Status_{fert}'
            if status_col in df.columns:
                compliance_stats[f'{fert}_status'] = {
                    'distribution': df[status_col].value_counts().to_dict(),
                    'overuse_rate': (df[status_col] == 'Overuse').mean() * 100,
                    'underuse_rate': (df[status_col] == 'Underuse').mean() * 100,
                    'normal_rate': (df[status_col] == 'Normal').mean() * 100
                }
        
        self.insights['compliance_analysis'] = compliance_stats
        
        # Print summary
        print(f"\n📊 Overall Compliance:")
        for status, count in compliance_stats['overall'].items():
            pct = compliance_stats['percentages'][status]
            emoji = "✅" if status == 'Normal' else "🟡" if status == 'Underuse' else "🔴"
            print(f"  {emoji} {status}: {count:,} ({pct:.1f}%)")
        
        # Compliance by fertilizer type
        print(f"\n📊 Compliance by Fertilizer Type:")
        for fert in ['Urea', 'NPK', 'Organik']:
            if f'{fert}_status' in compliance_stats:
                fert_stats = compliance_stats[f'{fert}_status']
                print(f"\n  {fert}:")
                print(f"    Normal: {fert_stats['normal_rate']:.1f}%")
                print(f"    Overuse: {fert_stats['overuse_rate']:.1f}%")
                print(f"    Underuse: {fert_stats['underuse_rate']:.1f}%")
        
        # Key insights
        print(f"\n💡 Key Insights:")
        
        overuse_rate = compliance_stats['percentages'].get('Overuse', 0)
        underuse_rate = compliance_stats['percentages'].get('Underuse', 0)
        
        if overuse_rate > underuse_rate:
            print(f"  ⚠️  Overuse is dominant ({overuse_rate:.1f}% vs {underuse_rate:.1f}%)")
            print(f"     → Focus on education about optimal dosage and efficiency")
            print(f"     → Implement monitoring on high-usage farmers")
        elif underuse_rate > overuse_rate:
            print(f"  ⚠️  Underuse is dominant ({underuse_rate:.1f}% vs {overuse_rate:.1f}%)")
            print(f"     → Focus on increasing access to fertilizers")
            print(f"     → Educate about benefits of optimal fertilizer use")
        else:
            print(f"  ✓  Balanced distribution ({overuse_rate:.1f}% overuse, {underuse_rate:.1f}% underuse)")
            print(f"     → Continue current monitoring approach")
        
        return compliance_stats
    
    def analyze_priority_recommendations(self, df):
        """Analisis prioritas rekomendasi"""
        print("\n" + "="*70)
        print("🎯 PRIORITY RECOMMENDATIONS ANALYSIS")
        print("="*70)
        
        if 'Prioritas' not in df.columns:
            print("⚠️  No priority information found")
            return None
        
        priority_stats = {
            'distribution': df['Prioritas'].value_counts().to_dict(),
            'percentages': (df['Prioritas'].value_counts() / len(df) * 100).to_dict()
        }
        
        # Action plan analysis
        if 'Action_Plan' in df.columns:
            action_counts = {}
            for actions in df['Action_Plan'].dropna():
                for action in str(actions).split(';'):
                    action = action.strip()
                    if action:
                        action_counts[action] = action_counts.get(action, 0) + 1
            
            priority_stats['top_actions'] = dict(sorted(
                action_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])
        
        self.insights['priority_analysis'] = priority_stats
        
        # Print summary
        print(f"\n📊 Priority Distribution:")
        for priority in ['Tinggi', 'Sedang', 'Rendah']:
            if priority in priority_stats['distribution']:
                count = priority_stats['distribution'][priority]
                pct = priority_stats['percentages'][priority]
                emoji = "🚨" if priority == 'Tinggi' else "⚠️" if priority == 'Sedang' else "✅"
                print(f"  {emoji} {priority}: {count:,} ({pct:.1f}%)")
        
        if 'top_actions' in priority_stats:
            print(f"\n📋 Top 10 Recommended Actions:")
            for i, (action, count) in enumerate(list(priority_stats['top_actions'].items())[:10], 1):
                print(f"  {i}. {action}: {count:,} farmers")
        
        # Strategic recommendations
        high_priority = priority_stats['distribution'].get('Tinggi', 0)
        high_pct = priority_stats['percentages'].get('Tinggi', 0)
        
        print(f"\n💡 Strategic Recommendations:")
        if high_pct > 20:
            print(f"  🚨 URGENT: {high_priority:,} farmers ({high_pct:.1f}%) need immediate action")
            print(f"     → Allocate resources for intensive field verification")
            print(f"     → Prioritize high-risk areas for intervention")
        elif high_pct > 10:
            print(f"  ⚠️  ATTENTION: {high_priority:,} farmers ({high_pct:.1f}%) require monitoring")
            print(f"     → Schedule targeted visits within 1-2 months")
            print(f"     → Provide educational materials")
        else:
            print(f"  ✓  STABLE: Only {high_priority:,} ({high_pct:.1f}%) need immediate action")
            print(f"     → Maintain regular monitoring schedule")
        
        return priority_stats
    
    def analyze_commodity_patterns(self, df):
        """Analisis pola per komoditas"""
        print("\n" + "="*70)
        print("🌾 COMMODITY-SPECIFIC ANALYSIS")
        print("="*70)
        
        if 'Komoditas' not in df.columns:
            print("⚠️  No commodity information found")
            return None
        
        commodity_stats = {}
        
        for commodity in sorted(df['Komoditas'].unique()):
            commodity_data = df[df['Komoditas'] == commodity]
            
            stats = {
                'count': len(commodity_data),
                'percentage': len(commodity_data) / len(df) * 100
            }
            
            # Land statistics
            if 'Luas_Tanah_ha' in commodity_data.columns:
                stats['land_mean'] = commodity_data['Luas_Tanah_ha'].mean()
                stats['land_total'] = commodity_data['Luas_Tanah_ha'].sum()
            
            # Fertilizer intensity
            if 'Total_per_ha' in commodity_data.columns:
                stats['intensity_mean'] = commodity_data['Total_per_ha'].mean()
                stats['intensity_median'] = commodity_data['Total_per_ha'].median()
            
            # Individual fertilizers
            for fert in ['Urea_per_ha', 'NPK_per_ha', 'Organik_per_ha']:
                if fert in commodity_data.columns:
                    stats[fert] = commodity_data[fert].mean()
            
            # Anomaly rate
            if 'Anomaly_Label' in commodity_data.columns:
                stats['anomaly_rate'] = (commodity_data['Anomaly_Label'] != 'Normal').mean() * 100
            
            # Compliance
            if 'Final_Status' in commodity_data.columns:
                stats['status_dist'] = commodity_data['Final_Status'].value_counts().to_dict()
                stats['overuse_rate'] = (commodity_data['Final_Status'] == 'Overuse').mean() * 100
                stats['underuse_rate'] = (commodity_data['Final_Status'] == 'Underuse').mean() * 100
            
            commodity_stats[commodity] = stats
        
        self.insights['commodity_analysis'] = commodity_stats
        
        # Print summary
        print(f"\n📊 Commodity Statistics:")
        for commodity, stats in commodity_stats.items():
            print(f"\n  {commodity}: {stats['count']:,} farmers ({stats['percentage']:.1f}%)")
            if 'land_mean' in stats:
                print(f"    └─ Average Land: {stats['land_mean']:.2f} ha")
            if 'intensity_mean' in stats:
                print(f"    └─ Average Intensity: {stats['intensity_mean']:.1f} kg/ha")
            if 'anomaly_rate' in stats:
                print(f"    └─ Anomaly Rate: {stats['anomaly_rate']:.1f}%")
            if 'overuse_rate' in stats and 'underuse_rate' in stats:
                print(f"    └─ Overuse: {stats['overuse_rate']:.1f}%, Underuse: {stats['underuse_rate']:.1f}%")
        
        return commodity_stats
    
    def generate_executive_summary(self):
        """Generate executive summary"""
        print("\n" + "="*70)
        print("📝 EXECUTIVE SUMMARY")
        print("="*70)
        
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'key_findings': []
        }
        
        # Key finding 1: Overall health
        if 'anomaly_analysis' in self.insights:
            anomaly_stats = self.insights['anomaly_analysis']
            total_anomalies = sum([v for k, v in anomaly_stats['distribution'].items() if k != 'Normal'])
            total_farmers = sum(anomaly_stats['distribution'].values())
            anomaly_rate = total_anomalies / total_farmers * 100
            
            summary['key_findings'].append({
                'category': 'System Health',
                'finding': f"Anomaly rate of {anomaly_rate:.1f}%",
                'severity': 'high' if anomaly_rate > 20 else 'medium' if anomaly_rate > 10 else 'low',
                'recommendation': 'Immediate field verification required' if anomaly_rate > 20 else 'Regular monitoring sufficient'
            })
        
        # Key finding 2: Compliance
        if 'compliance_analysis' in self.insights:
            compliance_stats = self.insights['compliance_analysis']
            if 'percentages' in compliance_stats:
                overuse_pct = compliance_stats['percentages'].get('Overuse', 0)
                underuse_pct = compliance_stats['percentages'].get('Underuse', 0)
                
                summary['key_findings'].append({
                    'category': 'Standards Compliance',
                    'finding': f"Overuse: {overuse_pct:.1f}%, Underuse: {underuse_pct:.1f}%",
                    'severity': 'high' if max(overuse_pct, underuse_pct) > 25 else 'medium',
                    'recommendation': 'Focus on education and monitoring'
                })
        
        # Key finding 3: Priority actions
        if 'priority_analysis' in self.insights:
            priority_stats = self.insights['priority_analysis']
            high_pct = priority_stats['percentages'].get('Tinggi', 0)
            
            summary['key_findings'].append({
                'category': 'Action Priority',
                'finding': f"{high_pct:.1f}% farmers require immediate action",
                'severity': 'high' if high_pct > 15 else 'medium' if high_pct > 5 else 'low',
                'recommendation': f"Allocate resources for {int(high_pct)}% of farmer base"
            })
        
        # Key finding 4: Clustering insights
        if 'clustering_analysis' in self.insights:
            cluster_stats = self.insights['clustering_analysis']
            n_clusters = cluster_stats['n_clusters']
            balance = cluster_stats['balance_score']
            
            summary['key_findings'].append({
                'category': 'Farmer Segmentation',
                'finding': f"{n_clusters} distinct farmer segments identified (balance: {balance:.1f}%)",
                'severity': 'low',
                'recommendation': 'Implement segment-specific strategies'
            })
        
        self.insights['executive_summary'] = summary
        
        # Print summary
        print(f"\n⏰ Analysis Date: {summary['timestamp']}")
        print(f"\n🔍 Key Findings:")
        
        for i, finding in enumerate(summary['key_findings'], 1):
            severity_emoji = "🚨" if finding['severity'] == 'high' else "⚠️" if finding['severity'] == 'medium' else "✅"
            print(f"\n  {i}. {finding['category']}: {severity_emoji}")
            print(f"     Finding: {finding['finding']}")
            print(f"     Action: {finding['recommendation']}")
        
        return summary
    
    def save_insights_report(self):
        """Save comprehensive insights to JSON file"""
        report_path = self.output_dir / "insights_report.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.insights, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Insights report saved: {report_path}")
        
        # Also save a human-readable summary
        summary_path = self.output_dir / "insights_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RDKK COMPREHENSIVE INSIGHTS REPORT\n")
            f.write("="*70 + "\n\n")
            
            if 'executive_summary' in self.insights:
                f.write(f"Analysis Date: {self.insights['executive_summary']['timestamp']}\n\n")
                
                f.write("KEY FINDINGS:\n")
                f.write("-" * 70 + "\n")
                for i, finding in enumerate(self.insights['executive_summary']['key_findings'], 1):
                    f.write(f"\n{i}. {finding['category']}\n")
                    f.write(f"   Finding: {finding['finding']}\n")
                    f.write(f"   Severity: {finding['severity'].upper()}\n")
                    f.write(f"   Recommendation: {finding['recommendation']}\n")
        
        print(f"💾 Summary report saved: {summary_path}")
    
    def run_full_analysis(self):
        """Run complete insight analysis"""
        print("\n" + "="*70)
        print("🚀 RDKK COMPREHENSIVE INSIGHT ANALYSIS")
        print("="*70)
        
        # Load data
        df = self.load_data()
        if df is None:
            print("✗ Cannot proceed without data")
            return
        
        model_info = self.load_model_info()
        
        # Run all analyses
        self.analyze_descriptive_statistics(df)
        self.analyze_anomaly_distribution(df)
        self.analyze_clustering_results(df)
        self.analyze_standards_compliance(df)
        self.analyze_priority_recommendations(df)
        self.analyze_commodity_patterns(df)
        self.generate_executive_summary()
        
        # Save reports
        self.save_insights_report()
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70)
        print("\n📁 Output Files:")
        print(f"  - {self.output_dir / 'insights_report.json'}")
        print(f"  - {self.output_dir / 'insights_summary.txt'}")
        print("\n🎯 Next Steps:")
        print("  1. Review insights_summary.txt for quick overview")
        print("  2. Check insights_report.json for detailed data")
        print("  3. Use dashboard (streamlit run app.py) for visual analysis")
        print("="*70 + "\n")


def main():
    """Main function"""
    analyzer = RDKKInsightAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()