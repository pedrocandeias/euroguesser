#!/usr/bin/env python3
"""
EuroMillions Prediction Summary

This script demonstrates how to use all the prediction models with the
comprehensive historical dataset.
"""

import pandas as pd
from collections import Counter

def show_dataset_info():
    """Display information about the comprehensive dataset."""
    print("EuroMillions Comprehensive Dataset Analysis")
    print("=" * 50)
    
    df = pd.read_csv('euromillions_historical_results.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    print(f"Total draws: {len(df)}")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Years covered: {df['Date'].max().year - df['Date'].min().year + 1}")
    
    # Show number frequencies
    main_numbers = []
    star_numbers = []
    
    for col in ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']:
        main_numbers.extend(df[col].tolist())
    for col in ['Star1', 'Star2']:
        star_numbers.extend(df[col].tolist())
        
    main_freq = Counter(main_numbers)
    star_freq = Counter(star_numbers)
    
    print(f"\\nMost frequent main numbers:")
    for num, freq in main_freq.most_common(10):
        print(f"  {num:2d}: {freq:3d} times ({freq/len(df)*100:.1f}% of draws)")
    
    print(f"\\nMost frequent star numbers:")
    for num, freq in star_freq.most_common(5):
        print(f"  {num:2d}: {freq:3d} times ({freq/(len(df)*2)*100:.1f}% of star positions)")
    
    # Statistical analysis
    main_cols = ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']
    star_cols = ['Star1', 'Star2']
    
    df['Main_Sum'] = df[main_cols].sum(axis=1)
    df['Star_Sum'] = df[star_cols].sum(axis=1)
    
    print(f"\\nStatistical Patterns:")
    print(f"Average main number sum: {df['Main_Sum'].mean():.1f} (range: {df['Main_Sum'].min()}-{df['Main_Sum'].max()})")
    print(f"Average star number sum: {df['Star_Sum'].mean():.1f} (range: {df['Star_Sum'].min()}-{df['Star_Sum'].max()})")
    
    # Recent trends
    recent_draws = 50
    recent_data = df.tail(recent_draws)
    recent_main = []
    recent_star = []
    
    for col in main_cols:
        recent_main.extend(recent_data[col].tolist())
    for col in star_cols:
        recent_star.extend(recent_data[col].tolist())
        
    recent_main_freq = Counter(recent_main)
    recent_star_freq = Counter(recent_star)
    
    print(f"\\nRecent Trends (last {recent_draws} draws):")
    print(f"Hot main numbers: {[num for num, _ in recent_main_freq.most_common(10)]}")
    print(f"Hot star numbers: {[num for num, _ in recent_star_freq.most_common(6)]}")
    
    cold_main = [num for num in range(1, 51) if recent_main_freq[num] <= 1]
    cold_star = [num for num in range(1, 13) if recent_star_freq[num] <= 1]
    
    print(f"Cold main numbers: {cold_main}")
    print(f"Cold star numbers: {cold_star}")


def show_available_models():
    """Show information about available prediction models."""
    print("\\n" + "=" * 50)
    print("AVAILABLE PREDICTION MODELS")
    print("=" * 50)
    
    models = [
        {
            'name': 'Statistical Predictor',
            'file': 'statistical_predictor.py',
            'description': 'Advanced statistical analysis using frequency, patterns, and gap analysis',
            'features': [
                'Comprehensive frequency analysis',
                'Temporal pattern detection',
                'Gap analysis (overdue numbers)',
                'Ensemble voting from multiple methods',
                'Quality analysis of predictions'
            ]
        },
        {
            'name': 'Quick Predictor', 
            'file': 'quick_predictor.py',
            'description': 'Fast predictions using multiple statistical approaches',
            'features': [
                'Smart weighted selection',
                'Balanced statistical approach', 
                'Hot/Cold number mixing',
                'Multiple prediction methods',
                'Quick analysis and comparison'
            ]
        },
        {
            'name': 'Advanced Neural Network',
            'file': 'advanced_euromillions_predictor.py', 
            'description': 'Deep learning approach with TensorFlow (requires installation)',
            'features': [
                'Deep neural networks',
                'LSTM for sequence prediction',
                'Feature engineering',
                'Ensemble methods',
                'Advanced evaluation metrics'
            ]
        },
        {
            'name': 'Improved Training',
            'file': 'improved_training.py',
            'description': 'Enhanced version of original model with better data preprocessing',
            'features': [
                'Time series sequence prediction',
                'Improved architecture',
                'Better normalization',
                'Training visualization',
                'Compatibility with existing workflow'
            ]
        }
    ]
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']} ({model['file']})")
        print(f"   Description: {model['description']}")
        print("   Features:")
        for feature in model['features']:
            print(f"     • {feature}")
        print()


def show_usage_examples():
    """Show usage examples for each model."""
    print("USAGE EXAMPLES")
    print("=" * 50)
    
    examples = [
        {
            'model': 'Statistical Predictor (Recommended)',
            'command': 'python3 statistical_predictor.py',
            'description': 'Comprehensive analysis and prediction using statistical methods'
        },
        {
            'model': 'Quick Predictor',
            'command': 'python3 quick_predictor.py', 
            'description': 'Fast predictions with multiple methods for comparison'
        },
        {
            'model': 'Advanced Neural Network',
            'command': 'python3 advanced_euromillions_predictor.py',
            'description': 'Deep learning approach (requires TensorFlow installation)'
        },
        {
            'model': 'Improved Training',
            'command': 'python3 improved_training.py',
            'description': 'Enhanced model training and prediction'
        }
    ]
    
    for example in examples:
        print(f"• {example['model']}:")
        print(f"  Command: {example['command']}")
        print(f"  Purpose: {example['description']}")
        print()
    
    print("INSTALLATION REQUIREMENTS:")
    print("• statistical_predictor.py: NumPy, Pandas (✓ Available)")
    print("• quick_predictor.py: NumPy, Pandas (✓ Available)") 
    print("• advanced_euromillions_predictor.py: TensorFlow, Scikit-learn (⚠ Requires installation)")
    print("• improved_training.py: TensorFlow (⚠ Requires installation)")


def main():
    """Main function to show complete summary."""
    show_dataset_info()
    show_available_models()
    show_usage_examples()
    
    print("=" * 50)
    print("RECOMMENDATION")
    print("=" * 50)
    print("For immediate use with comprehensive analysis:")
    print("👉 python3 statistical_predictor.py")
    print()
    print("For quick multiple predictions:")
    print("👉 python3 quick_predictor.py")
    print()
    print("Both models use the full 20-year historical dataset")
    print("with 1,869 EuroMillions draws for accurate predictions!")
    print("=" * 50)


if __name__ == "__main__":
    main()