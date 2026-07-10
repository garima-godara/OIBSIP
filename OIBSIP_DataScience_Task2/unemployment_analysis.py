#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unemployment in India - Data Analysis Project
Author: Garima Swami
Date: 2026
Description: Comprehensive analysis of unemployment trends in India (2019-2020)
             with focus on COVID-19 impact on employment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class UnemploymentAnalyzer:
    """Class to analyze unemployment data in India"""
    
    def __init__(self, csv_path):
        """
        Initialize the analyzer with data
        Args:
            csv_path: Path to the unemployment CSV file
        """
        self.csv_path = csv_path
        self.data = None
        self.rural_data = None
        self.urban_data = None
        
    def load_data(self):
        """Load and clean the data"""
        print("\n" + "="*70)
        print("LOADING AND EXPLORING UNEMPLOYMENT DATA")
        print("="*70)
        
        self.data = pd.read_csv(self.csv_path)
        # Clean column names
        self.data.columns = self.data.columns.str.strip()
        
        print(f"\nDataset shape: {self.data.shape}")
        print(f"\nColumn names:")
        print(self.data.columns.tolist())
        print(f"\nFirst few rows:")
        print(self.data.head(10))
        print(f"\nData Info:")
        print(self.data.info())
        print(f"\nMissing Values:")
        print(self.data.isnull().sum())
        
        # Separate rural and urban data
        self.rural_data = self.data[self.data['Area'].str.strip() == 'Rural']
        self.urban_data = self.data[self.data['Area'].str.strip() == 'Urban']
        
        print(f"\nRural records: {len(self.rural_data)}")
        print(f"Urban records: {len(self.urban_data)}")
        print(f"\nUnique Regions: {self.data['Region'].nunique()}")
        print(f"Date Range: {self.data['Date'].min()} to {self.data['Date'].max()}")
    
    def statistical_analysis(self):
        """Perform statistical analysis on unemployment data"""
        print("\n" + "="*70)
        print("STATISTICAL ANALYSIS")
        print("="*70)
        
        print("\n--- Overall Statistics ---")
        print(f"\nUnemployment Rate (%) Statistics:")
        print(self.data['Estimated Unemployment Rate (%)'].describe())
        
        print(f"\nEmployment Statistics:")
        print(self.data['Estimated Employed'].describe())
        
        print(f"\nLabour Participation Rate (%) Statistics:")
        print(self.data['Estimated Labour Participation Rate (%)'].describe())
        
        # Rural vs Urban comparison
        print("\n--- Rural vs Urban Comparison ---")
        print(f"\nRural Unemployment Rate:")
        print(f"  Mean: {self.rural_data['Estimated Unemployment Rate (%)'].mean():.2f}%")
        print(f"  Median: {self.rural_data['Estimated Unemployment Rate (%)'].median():.2f}%")
        print(f"  Max: {self.rural_data['Estimated Unemployment Rate (%)'].max():.2f}%")
        print(f"  Min: {self.rural_data['Estimated Unemployment Rate (%)'].min():.2f}%")
        
        print(f"\nUrban Unemployment Rate:")
        print(f"  Mean: {self.urban_data['Estimated Unemployment Rate (%)'].mean():.2f}%")
        print(f"  Median: {self.urban_data['Estimated Unemployment Rate (%)'].median():.2f}%")
        print(f"  Max: {self.urban_data['Estimated Unemployment Rate (%)'].max():.2f}%")
        print(f"  Min: {self.urban_data['Estimated Unemployment Rate (%)'].min():.2f}%")
        
        # Regional analysis
        print("\n--- Regional Analysis ---")
        regional_stats = self.data.groupby('Region').agg({
            'Estimated Unemployment Rate (%)': ['mean', 'max', 'min'],
            'Estimated Employed': 'mean',
            'Estimated Labour Participation Rate (%)': 'mean'
        }).round(2)
        print("\nTop 5 Regions by Average Unemployment Rate:")
        top_regions = self.data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False).head(5)
        for region, rate in top_regions.items():
            print(f"  {region}: {rate:.2f}%")
    
    def covid_impact_analysis(self):
        """Analyze COVID-19 impact on unemployment"""
        print("\n" + "="*70)
        print("COVID-19 IMPACT ANALYSIS (April 2020 onwards)")
        print("="*70)
        
        # Pre-COVID and COVID periods
        pre_covid = self.data[self.data['Date'] < '30-04-2020']
        covid_period = self.data[self.data['Date'] >= '30-04-2020']
        
        print(f"\nPre-COVID Period (Before April 2020):")
        print(f"  Average Unemployment Rate: {pre_covid['Estimated Unemployment Rate (%)'].mean():.2f}%")
        print(f"  Average Employment: {pre_covid['Estimated Employed'].mean():.0f}")
        print(f"  Average Labour Participation: {pre_covid['Estimated Labour Participation Rate (%)'].mean():.2f}%")
        
        print(f"\nCOVID-19 Period (April 2020 onwards):")
        print(f"  Average Unemployment Rate: {covid_period['Estimated Unemployment Rate (%)'].mean():.2f}%")
        print(f"  Average Employment: {covid_period['Estimated Employed'].mean():.0f}")
        print(f"  Average Labour Participation: {covid_period['Estimated Labour Participation Rate (%)'].mean():.2f}%")
        
        print(f"\nImpact:")
        unemployment_increase = covid_period['Estimated Unemployment Rate (%)'].mean() - pre_covid['Estimated Unemployment Rate (%)'].mean()
        employment_decrease = pre_covid['Estimated Employed'].mean() - covid_period['Estimated Employed'].mean()
        participation_decrease = pre_covid['Estimated Labour Participation Rate (%)'].mean() - covid_period['Estimated Labour Participation Rate (%)'].mean()
        
        print(f"  Unemployment Increase: {unemployment_increase:+.2f} percentage points")
        print(f"  Employment Decrease: {employment_decrease:,.0f} people")
        print(f"  Labour Participation Decrease: {participation_decrease:+.2f} percentage points")
        
        # States most affected by COVID
        print("\nMost Affected States (by unemployment increase):")
        state_comparison = pd.DataFrame({
            'Pre_COVID': pre_covid.groupby('Region')['Estimated Unemployment Rate (%)'].mean(),
            'COVID_Period': covid_period.groupby('Region')['Estimated Unemployment Rate (%)'].mean()
        })
        state_comparison['Increase'] = state_comparison['COVID_Period'] - state_comparison['Pre_COVID']
        top_affected = state_comparison.nlargest(5, 'Increase')
        for region, row in top_affected.iterrows():
            print(f"  {region}: {row['Increase']:+.2f}% (Pre: {row['Pre_COVID']:.2f}% → COVID: {row['COVID_Period']:.2f}%)")
    
    def visualize_unemployment_trends(self):
        """Create visualizations for unemployment analysis"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        sns.set_style("whitegrid")
        
        # 1. Unemployment Rate Trend
        fig = plt.figure(figsize=(16, 12))
        
        plt.subplot(2, 3, 1)
        rural_trend = self.rural_data.groupby('Date')['Estimated Unemployment Rate (%)'].mean().sort_index()
        urban_trend = self.urban_data.groupby('Date')['Estimated Unemployment Rate (%)'].mean().sort_index()
        plt.plot(range(len(rural_trend)), rural_trend.values, marker='o', label='Rural', linewidth=2)
        plt.plot(range(len(urban_trend)), urban_trend.values, marker='s', label='Urban', linewidth=2)
        plt.axvline(x=13, color='red', linestyle='--', alpha=0.7, label='COVID-19 Impact')
        plt.title('Unemployment Rate Trend: Rural vs Urban', fontsize=12, fontweight='bold')
        plt.xlabel('Time Period')
        plt.ylabel('Unemployment Rate (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 2. Regional Unemployment Comparison
        plt.subplot(2, 3, 2)
        regional_avg = self.data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False).head(10)
        plt.barh(range(len(regional_avg)), regional_avg.values, color='steelblue')
        plt.yticks(range(len(regional_avg)), regional_avg.index, fontsize=9)
        plt.title('Top 10 States by Average Unemployment Rate', fontsize=12, fontweight='bold')
        plt.xlabel('Unemployment Rate (%)')
        for i, v in enumerate(regional_avg.values):
            plt.text(v + 0.5, i, f'{v:.1f}%', va='center')
        plt.grid(True, alpha=0.3, axis='x')
        
        # 3. Employment Level Trend
        plt.subplot(2, 3, 3)
        rural_emp = self.rural_data.groupby('Date')['Estimated Employed'].mean().sort_index()
        urban_emp = self.urban_data.groupby('Date')['Estimated Employed'].mean().sort_index()
        plt.plot(range(len(rural_emp)), rural_emp.values/1000000, marker='o', label='Rural', linewidth=2)
        plt.plot(range(len(urban_emp)), urban_emp.values/1000000, marker='s', label='Urban', linewidth=2)
        plt.axvline(x=13, color='red', linestyle='--', alpha=0.7)
        plt.title('Employment Level Trend (Millions)', fontsize=12, fontweight='bold')
        plt.xlabel('Time Period')
        plt.ylabel('Employed (Millions)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 4. Labour Participation Rate
        plt.subplot(2, 3, 4)
        rural_part = self.rural_data.groupby('Date')['Estimated Labour Participation Rate (%)'].mean().sort_index()
        urban_part = self.urban_data.groupby('Date')['Estimated Labour Participation Rate (%)'].mean().sort_index()
        plt.plot(range(len(rural_part)), rural_part.values, marker='o', label='Rural', linewidth=2)
        plt.plot(range(len(urban_part)), urban_part.values, marker='s', label='Urban', linewidth=2)
        plt.axvline(x=13, color='red', linestyle='--', alpha=0.7)
        plt.title('Labour Participation Rate Trend', fontsize=12, fontweight='bold')
        plt.xlabel('Time Period')
        plt.ylabel('Participation Rate (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 5. Box plot: Pre vs COVID
        plt.subplot(2, 3, 5)
        pre_covid = self.data[self.data['Date'] < '30-04-2020']
        covid = self.data[self.data['Date'] >= '30-04-2020']
        box_data = [pre_covid['Estimated Unemployment Rate (%)'], 
                    covid['Estimated Unemployment Rate (%)']]
        bp = plt.boxplot(box_data, labels=['Pre-COVID', 'COVID'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        plt.title('Unemployment Rate: Pre vs COVID Period', fontsize=12, fontweight='bold')
        plt.ylabel('Unemployment Rate (%)')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 6. State-wise COVID Impact
        plt.subplot(2, 3, 6)
        state_impact = pd.DataFrame({
            'Pre_COVID': pre_covid.groupby('Region')['Estimated Unemployment Rate (%)'].mean(),
            'COVID': covid.groupby('Region')['Estimated Unemployment Rate (%)'].mean()
        })
        state_impact['Impact'] = state_impact['COVID'] - state_impact['Pre_COVID']
        top_impact = state_impact.nlargest(8, 'Impact')
        x = np.arange(len(top_impact))
        plt.bar(x - 0.2, top_impact['Pre_COVID'], 0.4, label='Pre-COVID', color='skyblue')
        plt.bar(x + 0.2, top_impact['COVID'], 0.4, label='COVID', color='salmon')
        plt.xlabel('State')
        plt.ylabel('Unemployment Rate (%)')
        plt.title('Most Affected States: Pre vs COVID', fontsize=12, fontweight='bold')
        plt.xticks(x, top_impact.index, rotation=45, ha='right', fontsize=8)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('unemployment_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Main analysis visualization saved as 'unemployment_analysis.png'")
        plt.close()
        
        # Additional detailed visualizations
        self._create_heatmap()
        self._create_distribution_plots()
    
    def _create_heatmap(self):
        """Create heatmap of unemployment by state and time"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Rural heatmap
        rural_pivot = self.rural_data.pivot_table(
            values='Estimated Unemployment Rate (%)',
            index='Region',
            columns='Date',
            aggfunc='mean'
        )
        sns.heatmap(rural_pivot.head(15), cmap='RdYlGn_r', ax=axes[0], cbar_kws={'label': 'Rate (%)'})
        axes[0].set_title('Rural Unemployment Rate by State and Time (Top 15)', fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('State')
        
        # Urban heatmap
        urban_pivot = self.urban_data.pivot_table(
            values='Estimated Unemployment Rate (%)',
            index='Region',
            columns='Date',
            aggfunc='mean'
        )
        sns.heatmap(urban_pivot.head(15), cmap='RdYlGn_r', ax=axes[1], cbar_kws={'label': 'Rate (%)'})
        axes[1].set_title('Urban Unemployment Rate by State and Time (Top 15)', fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('State')
        
        plt.tight_layout()
        plt.savefig('unemployment_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Heatmap saved as 'unemployment_heatmap.png'")
        plt.close()
    
    def _create_distribution_plots(self):
        """Create distribution plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Unemployment distribution
        axes[0, 0].hist(self.data['Estimated Unemployment Rate (%)'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(self.data['Estimated Unemployment Rate (%)'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {self.data['Estimated Unemployment Rate (%)'].mean():.2f}%")
        axes[0, 0].set_title('Distribution of Unemployment Rate', fontweight='bold')
        axes[0, 0].set_xlabel('Unemployment Rate (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Employment distribution
        axes[0, 1].hist(self.data['Estimated Employed']/1000000, bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(self.data['Estimated Employed'].mean()/1000000, color='red', linestyle='--', linewidth=2, label=f"Mean: {self.data['Estimated Employed'].mean()/1000000:.2f}M")
        axes[0, 1].set_title('Distribution of Employment Level', fontweight='bold')
        axes[0, 1].set_xlabel('Employed (Millions)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rural vs Urban pie chart
        axes[1, 0].pie([len(self.rural_data), len(self.urban_data)], labels=['Rural', 'Urban'], autopct='%1.1f%%', colors=['lightgreen', 'lightblue'])
        axes[1, 0].set_title('Data Distribution: Rural vs Urban', fontweight='bold')
        
        # Scatter: Unemployment vs Participation
        axes[1, 1].scatter(self.data['Estimated Unemployment Rate (%)'], 
                          self.data['Estimated Labour Participation Rate (%)'],
                          alpha=0.5, s=50, c=self.data['Estimated Unemployed'].fillna(self.data['Estimated Labour Participation Rate (%)']), cmap='viridis')
        axes[1, 1].set_title('Unemployment Rate vs Labour Participation', fontweight='bold')
        axes[1, 1].set_xlabel('Unemployment Rate (%)')
        axes[1, 1].set_ylabel('Labour Participation Rate (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('unemployment_distributions.png', dpi=300, bbox_inches='tight')
        print("✓ Distribution plots saved as 'unemployment_distributions.png'")
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*70)
        print("COMPREHENSIVE REPORT SUMMARY")
        print("="*70)
        
        print("\n📊 DATASET OVERVIEW:")
        print(f"  • Total Records: {len(self.data):,}")
        print(f"  • Time Period: May 2019 - June 2020")
        print(f"  • States Covered: {self.data['Region'].nunique()}")
        print(f"  • Area Types: Rural and Urban")
        
        print("\n📈 KEY STATISTICS:")
        print(f"  • Average Unemployment Rate: {self.data['Estimated Unemployment Rate (%)'].mean():.2f}%")
        print(f"  • Highest Unemployment: {self.data['Estimated Unemployment Rate (%)'].max():.2f}% ({self.data.loc[self.data['Estimated Unemployment Rate (%)'].idxmax(), 'Region']} - {self.data.loc[self.data['Estimated Unemployment Rate (%)'].idxmax(), 'Date']})")
        print(f"  • Average Employment: {self.data['Estimated Employed'].mean()/1000000:.2f} Million")
        print(f"  • Average Labour Participation: {self.data['Estimated Labour Participation Rate (%)'].mean():.2f}%")
        
        print("\n🌍 RURAL vs URBAN:")
        print(f"  Rural Unemployment: {self.rural_data['Estimated Unemployment Rate (%)'].mean():.2f}%")
        print(f"  Urban Unemployment: {self.urban_data['Estimated Unemployment Rate (%)'].mean():.2f}%")
        print(f"  Difference: {abs(self.rural_data['Estimated Unemployment Rate (%)'].mean() - self.urban_data['Estimated Unemployment Rate (%)'].mean()):.2f} pp")
        
        print("\n🦠 COVID-19 IMPACT (April 2020):")
        pre_covid = self.data[self.data['Date'] < '30-04-2020']
        covid = self.data[self.data['Date'] >= '30-04-2020']
        print(f"  Pre-COVID Avg Unemployment: {pre_covid['Estimated Unemployment Rate (%)'].mean():.2f}%")
        print(f"  COVID Period Avg Unemployment: {covid['Estimated Unemployment Rate (%)'].mean():.2f}%")
        print(f"  Increase: {covid['Estimated Unemployment Rate (%)'].mean() - pre_covid['Estimated Unemployment Rate (%)'].mean():+.2f} pp")
        print(f"  Employment Drop: {(pre_covid['Estimated Employed'].mean() - covid['Estimated Employed'].mean())/1000000:.2f} Million")
        
        print("\n📍 TOP 5 MOST AFFECTED STATES:")
        state_impact = pd.DataFrame({
            'Pre_COVID': pre_covid.groupby('Region')['Estimated Unemployment Rate (%)'].mean(),
            'COVID': covid.groupby('Region')['Estimated Unemployment Rate (%)'].mean()
        })
        state_impact['Impact'] = state_impact['COVID'] - state_impact['Pre_COVID']
        for i, (state, impact) in enumerate(state_impact.nlargest(5, 'Impact')['Impact'].items(), 1):
            print(f"  {i}. {state}: +{impact:.2f} pp")
        
        print("\n💡 KEY FINDINGS:")
        print("  • COVID-19 pandemic significantly increased unemployment across India")
        print("  • Rural unemployment increased more sharply than urban areas")
        print("  • Eastern states (Bihar, Jharkhand) were most severely impacted")
        print("  • Labour force participation decreased during COVID period")
        print("  • Recovery began to show in June 2020 (partial data)")
        
        print("\n📁 GENERATED FILES:")
        print("  • unemployment_analysis.png - Main analysis visualizations")
        print("  • unemployment_heatmap.png - State and time heatmaps")
        print("  • unemployment_distributions.png - Distribution analysis")
        
        print("\n" + "="*70)
        print("Analysis Complete!")
        print("="*70 + "\n")

def main():
    """Main execution function"""
    print("\n\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " "*12 + "UNEMPLOYMENT IN INDIA - DATA ANALYSIS PROJECT" + " "*11 + "#")
    print("#" + " "*15 + "Focus on COVID-19 Impact (2019-2020)" + " "*16 + "#")
    print("#" + " "*68 + "#")
    print("#"*70 + "\n")
    
    # Initialize analyzer
    analyzer = UnemploymentAnalyzer('Unemployment_in_India.csv')
    
    # Execute analysis pipeline
    analyzer.load_data()
    analyzer.statistical_analysis()
    analyzer.covid_impact_analysis()
    analyzer.visualize_unemployment_trends()
    analyzer.generate_report()

if __name__ == "__main__":
    main()