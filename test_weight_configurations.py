#!/usr/bin/env python3
"""
Test script to demonstrate different weight configurations
for the Statistical EuroMillions Prediction System.

This script shows how different analysis weight combinations
affect the prediction outcomes.
"""

import numpy as np
from statistical_predictor import StatisticalEuroMillionsPredictor


def test_configuration(name, frequency_weight, gap_weight, pattern_weight, temporal_weight):
    """Test a specific weight configuration."""
    print(f"\n{'='*80}")
    print(f"TESTING: {name}")
    print(f"{'='*80}")
    
    # Create predictor with custom weights
    predictor = StatisticalEuroMillionsPredictor(
        frequency_weight=frequency_weight,
        gap_weight=gap_weight,
        pattern_weight=pattern_weight,
        temporal_weight=temporal_weight
    )
    
    # Load and analyze data
    predictor.load_and_analyze_data()
    
    # Generate prediction
    predicted_main, predicted_stars = predictor.generate_ensemble_prediction()
    
    print(f"\n🎯 PREDICTION RESULT:")
    print(f"Main Numbers: {predicted_main}")
    print(f"Star Numbers: {predicted_stars}")
    
    return predicted_main, predicted_stars


def main():
    """Test multiple weight configurations."""
    print("EuroMillions Predictor - Weight Configuration Testing")
    print("=" * 60)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Test different configurations
    configurations = [
        ("Default (Balanced)", 25.0, 25.0, 25.0, 25.0),
        ("Frequency Focused", 60.0, 15.0, 15.0, 10.0),
        ("Gap Analysis Focused", 10.0, 70.0, 15.0, 5.0),
        ("Pattern Analysis Focused", 15.0, 10.0, 65.0, 10.0),
        ("Temporal Analysis Focused", 10.0, 10.0, 20.0, 60.0),
        ("Frequency + Gap Only", 50.0, 50.0, 0.0, 0.0),
        ("Pattern Only", 0.0, 0.0, 100.0, 0.0),
        ("Hot Numbers (Recent Trends)", 40.0, 40.0, 10.0, 10.0),
    ]
    
    results = {}
    
    for name, freq, gap, pattern, temporal in configurations:
        main_nums, star_nums = test_configuration(name, freq, gap, pattern, temporal)
        results[name] = (main_nums, star_nums)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL CONFIGURATIONS")
    print(f"{'='*80}")
    
    for name, (main_nums, star_nums) in results.items():
        print(f"{name:25} | Main: {main_nums} | Stars: {star_nums}")
    
    # Analyze number frequency across predictions
    print(f"\n{'='*80}")
    print("NUMBER FREQUENCY ANALYSIS ACROSS CONFIGURATIONS")
    print(f"{'='*80}")
    
    from collections import Counter
    all_main_numbers = []
    all_star_numbers = []
    
    for main_nums, star_nums in results.values():
        all_main_numbers.extend(main_nums)
        all_star_numbers.extend(star_nums)
    
    main_freq = Counter(all_main_numbers)
    star_freq = Counter(all_star_numbers)
    
    print(f"Most frequent main numbers across all predictions: {main_freq.most_common(10)}")
    print(f"Most frequent star numbers across all predictions: {star_freq.most_common(5)}")
    
    print(f"\n🎯 Testing completed! Each configuration produces different predictions")
    print(f"   based on the weighted combination of analysis methods.")


if __name__ == "__main__":
    main()