#!/usr/bin/env python3
"""
Quick EuroMillions Predictor

A simple prediction system that works with the comprehensive historical dataset
and generates predictions using multiple statistical approaches.

Compatible with your existing workflow.
"""

import numpy as np
import pandas as pd
from collections import Counter
import random


def load_historical_data(csv_path='euromillions_historical_results.csv'):
    """Load the comprehensive historical dataset."""
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def analyze_frequencies(df):
    """Analyze number frequencies."""
    # Main numbers
    main_numbers = []
    for col in ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']:
        main_numbers.extend(df[col].tolist())
    main_freq = Counter(main_numbers)
    
    # Star numbers
    star_numbers = []
    for col in ['Star1', 'Star2']:
        star_numbers.extend(df[col].tolist())
    star_freq = Counter(star_numbers)
    
    return main_freq, star_freq


def get_recent_trends(df, recent_draws=30):
    """Analyze recent trends."""
    if len(df) < recent_draws:
        recent_draws = len(df)
        
    recent_data = df.tail(recent_draws)
    
    recent_main = []
    recent_stars = []
    
    for col in ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']:
        recent_main.extend(recent_data[col].tolist())
    for col in ['Star1', 'Star2']:
        recent_stars.extend(recent_data[col].tolist())
        
    recent_main_freq = Counter(recent_main)
    recent_star_freq = Counter(recent_stars)
    
    return recent_main_freq, recent_star_freq


def weighted_selection(candidates, weights, count):
    """Select numbers using weighted probabilities."""
    if len(candidates) < count:
        return candidates
        
    # Normalize weights
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    
    # Select without replacement
    selected = []
    remaining_candidates = candidates.copy()
    remaining_probs = probabilities.copy()
    
    for _ in range(count):
        if not remaining_candidates:
            break
            
        # Renormalize probabilities
        total_remaining = sum(remaining_probs)
        if total_remaining == 0:
            break
            
        normalized_probs = [p / total_remaining for p in remaining_probs]
        
        # Select based on probability
        selected_idx = np.random.choice(len(remaining_candidates), p=normalized_probs)
        selected.append(remaining_candidates[selected_idx])
        
        # Remove selected
        remaining_candidates.pop(selected_idx)
        remaining_probs.pop(selected_idx)
    
    return selected


def predict_smart_selection(df):
    """Generate prediction using smart selection algorithm."""
    print("Generating smart prediction...")
    
    # Analyze frequencies
    main_freq, star_freq = analyze_frequencies(df)
    recent_main_freq, recent_star_freq = get_recent_trends(df, 30)
    
    # Create weighted selection for main numbers
    main_candidates = list(range(1, 51))
    main_weights = []
    
    for num in main_candidates:
        weight = main_freq.get(num, 0)  # Historical frequency
        weight += recent_main_freq.get(num, 0) * 2  # Recent frequency (boosted)
        
        # Small boost for numbers that haven't appeared recently
        if recent_main_freq.get(num, 0) == 0:
            weight += 10
            
        main_weights.append(weight)
    
    # Create weighted selection for star numbers
    star_candidates = list(range(1, 13))
    star_weights = []
    
    for num in star_candidates:
        weight = star_freq.get(num, 0)
        weight += recent_star_freq.get(num, 0) * 2
        
        if recent_star_freq.get(num, 0) == 0:
            weight += 5
            
        star_weights.append(weight)
    
    # Select numbers
    main_prediction = weighted_selection(main_candidates, main_weights, 5)
    star_prediction = weighted_selection(star_candidates, star_weights, 2)
    
    return sorted(main_prediction), sorted(star_prediction)


def predict_balanced_approach(df):
    """Generate prediction using balanced statistical approach."""
    print("Generating balanced prediction...")
    
    main_freq, star_freq = analyze_frequencies(df)
    
    # Calculate statistics
    main_cols = ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']
    df['MainSum'] = df[main_cols].sum(axis=1)
    avg_sum = df['MainSum'].mean()
    
    # Generate many candidates and select best one
    best_main = None
    best_stars = None
    best_score = float('inf')
    
    for _ in range(500):  # Try 500 combinations
        # Generate candidate
        main_nums = sorted(np.random.choice(range(1, 51), 5, replace=False))
        star_nums = sorted(np.random.choice(range(1, 13), 2, replace=False))
        
        # Score based on multiple criteria
        score = 0
        
        # 1. Sum similarity to historical average
        main_sum = sum(main_nums)
        score += abs(main_sum - avg_sum)
        
        # 2. Frequency score (prefer frequently drawn numbers)
        freq_score = sum(main_freq.get(num, 0) for num in main_nums)
        freq_score += sum(star_freq.get(num, 0) for num in star_nums) * 5  # Weight stars more
        score -= freq_score * 0.1  # Lower score is better
        
        # 3. Distribution score (prefer spread across range)
        if max(main_nums) - min(main_nums) > 25:  # Good spread
            score -= 10
            
        # 4. Even/odd balance
        even_count = sum(1 for n in main_nums if n % 2 == 0)
        if even_count in [2, 3]:  # Balanced even/odd
            score -= 5
            
        if score < best_score:
            best_score = score
            best_main = main_nums
            best_stars = star_nums
    
    return best_main, best_stars


def predict_hot_cold_mix(df):
    """Generate prediction mixing hot and cold numbers."""
    print("Generating hot/cold mix prediction...")
    
    recent_main_freq, recent_star_freq = get_recent_trends(df, 40)
    
    # Identify hot and cold numbers
    hot_main = [num for num, freq in recent_main_freq.most_common(15)]
    cold_main = [num for num in range(1, 51) if recent_main_freq.get(num, 0) <= 1]
    
    hot_stars = [num for num, freq in recent_star_freq.most_common(6)]
    cold_stars = [num for num in range(1, 13) if recent_star_freq.get(num, 0) <= 1]
    
    # Mix hot and cold
    main_prediction = []
    
    # Select 3 hot numbers
    if len(hot_main) >= 3:
        main_prediction.extend(random.sample(hot_main[:10], 3))
    
    # Select 2 cold numbers
    if len(cold_main) >= 2:
        main_prediction.extend(random.sample(cold_main, min(2, len(cold_main))))
    
    # Fill remaining with medium frequency
    remaining_main = [n for n in range(1, 51) if n not in main_prediction and n not in hot_main[:5] and n not in cold_main[:5]]
    while len(main_prediction) < 5 and remaining_main:
        main_prediction.append(remaining_main.pop(random.randint(0, len(remaining_main)-1)))
    
    # Similar for stars
    star_prediction = []
    if len(hot_stars) >= 1:
        star_prediction.append(random.choice(hot_stars[:4]))
    if len(cold_stars) >= 1:
        star_prediction.append(random.choice(cold_stars))
        
    # Fill if needed
    while len(star_prediction) < 2:
        remaining_stars = [n for n in range(1, 13) if n not in star_prediction]
        star_prediction.append(random.choice(remaining_stars))
    
    return sorted(main_prediction), sorted(star_prediction)


def generate_multiple_predictions():
    """Generate multiple predictions using different methods."""
    print("EuroMillions Quick Prediction System")
    print("=" * 40)
    
    # Load data
    df = load_historical_data()
    print(f"Loaded {len(df)} historical draws")
    
    # Generate predictions using different methods
    print(f"\\nGenerating predictions using multiple methods...")
    
    # Set seed for reproducible results (remove for random predictions)
    np.random.seed(None)  # Use None for truly random
    random.seed(None)
    
    # Method 1: Smart weighted selection
    smart_main, smart_stars = predict_smart_selection(df)
    
    # Method 2: Balanced statistical approach
    balanced_main, balanced_stars = predict_balanced_approach(df)
    
    # Method 3: Hot/Cold mix
    hotcold_main, hotcold_stars = predict_hot_cold_mix(df)
    
    # Display results
    print(f"\\n" + "="*40)
    print("PREDICTION RESULTS")
    print("="*40)
    print(f"Method 1 (Smart Selection):  Main {smart_main}, Stars {smart_stars}")
    print(f"Method 2 (Balanced Stats):   Main {balanced_main}, Stars {balanced_stars}")
    print(f"Method 3 (Hot/Cold Mix):     Main {hotcold_main}, Stars {hotcold_stars}")
    
    # Show some statistics
    main_freq, star_freq = analyze_frequencies(df)
    recent_main, recent_star = get_recent_trends(df, 30)
    
    print(f"\\nDataset Statistics:")
    print(f"Most frequent main numbers: {[num for num, _ in main_freq.most_common(10)]}")
    print(f"Most frequent star numbers: {[num for num, _ in star_freq.most_common(5)]}")
    print(f"Hot main numbers (last 30): {[num for num, _ in recent_main.most_common(10)]}")
    print(f"Hot star numbers (last 30): {[num for num, _ in recent_star.most_common(5)]}")
    
    # Quick analysis of predictions
    print(f"\\nPrediction Analysis:")
    for method, (main, stars) in [("Smart", (smart_main, smart_stars)), 
                                  ("Balanced", (balanced_main, balanced_stars)), 
                                  ("Hot/Cold", (hotcold_main, hotcold_stars))]:
        main_sum = sum(main)
        even_count = sum(1 for n in main if n % 2 == 0)
        print(f"{method:>8}: Sum={main_sum:>3}, Even/Odd={even_count}/{5-even_count}, Range={max(main)-min(main):>2}")
    
    return {
        'smart': (smart_main, smart_stars),
        'balanced': (balanced_main, balanced_stars),
        'hotcold': (hotcold_main, hotcold_stars)
    }


if __name__ == "__main__":
    try:
        predictions = generate_multiple_predictions()
        print(f"\\n{'='*40}")
        print("Good luck! 🍀")
        print("="*40)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()