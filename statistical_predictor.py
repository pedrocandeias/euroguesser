#!/usr/bin/env python3
"""
Statistical EuroMillions Prediction System

This script implements statistical analysis and prediction methods for EuroMillions
using only NumPy and Pandas, focusing on frequency analysis, pattern detection,
and probabilistic modeling.

Author: Claude Code Assistant
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import random


class StatisticalEuroMillionsPredictor:
    def __init__(self, data_path='euromillions_historical_results.csv'):
        """Initialize the predictor with historical data."""
        self.data_path = data_path
        self.df = None
        self.main_frequencies = {}
        self.star_frequencies = {}
        self.patterns = {}
        
    def load_and_analyze_data(self):
        """Load the dataset and perform comprehensive statistical analysis."""
        print("Loading and analyzing comprehensive dataset...")
        self.df = pd.read_csv(self.data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        print(f"Dataset loaded: {len(self.df)} draws from {self.df['Date'].min()} to {self.df['Date'].max()}")
        
        # Perform comprehensive analysis
        self.analyze_frequencies()
        self.analyze_temporal_patterns()
        self.analyze_number_patterns()
        self.analyze_recent_trends()
        
    def analyze_frequencies(self):
        """Analyze frequency distribution of all numbers."""
        print("\\nAnalyzing number frequencies...")
        
        # Main numbers frequency
        main_numbers = []
        for col in ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']:
            main_numbers.extend(self.df[col].tolist())
        
        self.main_frequencies = Counter(main_numbers)
        
        # Star numbers frequency
        star_numbers = []
        for col in ['Star1', 'Star2']:
            star_numbers.extend(self.df[col].tolist())
            
        self.star_frequencies = Counter(star_numbers)
        
        # Print most/least frequent numbers
        most_frequent_main = self.main_frequencies.most_common(10)
        least_frequent_main = self.main_frequencies.most_common()[-10:]
        most_frequent_star = self.star_frequencies.most_common(5)
        least_frequent_star = self.star_frequencies.most_common()[-5:]
        
        print(f"Most frequent main numbers: {most_frequent_main}")
        print(f"Least frequent main numbers: {least_frequent_main}")
        print(f"Most frequent star numbers: {most_frequent_star}")
        print(f"Least frequent star numbers: {least_frequent_star}")
        
    def analyze_temporal_patterns(self):
        """Analyze patterns based on time periods."""
        print("\\nAnalyzing temporal patterns...")
        
        # Add temporal features
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
        self.df['Quarter'] = self.df['Date'].dt.quarter
        
        # Analyze by year
        yearly_stats = {}
        for year in self.df['Year'].unique():
            year_data = self.df[self.df['Year'] == year]
            yearly_stats[year] = {
                'count': len(year_data),
                'avg_main_sum': year_data[['Main1', 'Main2', 'Main3', 'Main4', 'Main5']].sum(axis=1).mean(),
                'avg_star_sum': year_data[['Star1', 'Star2']].sum(axis=1).mean()
            }
        
        self.patterns['yearly'] = yearly_stats
        
        # Analyze by month
        monthly_main_freq = {}
        monthly_star_freq = {}
        
        for month in range(1, 13):
            month_data = self.df[self.df['Month'] == month]
            main_nums = []
            star_nums = []
            
            for col in ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']:
                main_nums.extend(month_data[col].tolist())
            for col in ['Star1', 'Star2']:
                star_nums.extend(month_data[col].tolist())
                
            monthly_main_freq[month] = Counter(main_nums)
            monthly_star_freq[month] = Counter(star_nums)
        
        self.patterns['monthly_main'] = monthly_main_freq
        self.patterns['monthly_star'] = monthly_star_freq
        
        print("Temporal analysis completed")
        
    def analyze_number_patterns(self):
        """Analyze mathematical patterns in the numbers."""
        print("\\nAnalyzing number patterns...")
        
        # Calculate sums, ranges, and patterns for each draw
        main_cols = ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']
        star_cols = ['Star1', 'Star2']
        
        self.df['Main_Sum'] = self.df[main_cols].sum(axis=1)
        self.df['Main_Range'] = self.df[main_cols].max(axis=1) - self.df[main_cols].min(axis=1)
        self.df['Star_Sum'] = self.df[star_cols].sum(axis=1)
        
        # Count even/odd numbers
        main_even_counts = []
        main_odd_counts = []
        
        for idx, row in self.df.iterrows():
            main_numbers = [row[col] for col in main_cols]
            even_count = sum(1 for n in main_numbers if n % 2 == 0)
            odd_count = 5 - even_count
            main_even_counts.append(even_count)
            main_odd_counts.append(odd_count)
            
        self.df['Main_Even_Count'] = main_even_counts
        self.df['Main_Odd_Count'] = main_odd_counts
        
        # Analyze consecutive numbers
        consecutive_counts = []
        for idx, row in self.df.iterrows():
            main_numbers = sorted([row[col] for col in main_cols])
            consecutive = 0
            for i in range(len(main_numbers) - 1):
                if main_numbers[i+1] - main_numbers[i] == 1:
                    consecutive += 1
            consecutive_counts.append(consecutive)
            
        self.df['Consecutive_Count'] = consecutive_counts
        
        # Statistics
        print(f"Average main sum: {self.df['Main_Sum'].mean():.2f} (std: {self.df['Main_Sum'].std():.2f})")
        print(f"Average main range: {self.df['Main_Range'].mean():.2f}")
        print(f"Average star sum: {self.df['Star_Sum'].mean():.2f} (std: {self.df['Star_Sum'].std():.2f})")
        print(f"Average even/odd distribution: {self.df['Main_Even_Count'].mean():.2f} even, {self.df['Main_Odd_Count'].mean():.2f} odd")
        print(f"Average consecutive numbers: {self.df['Consecutive_Count'].mean():.2f}")
        
    def analyze_recent_trends(self, recent_draws=50):
        """Analyze trends in the most recent draws."""
        print(f"\\nAnalyzing recent trends (last {recent_draws} draws)...")
        
        if len(self.df) < recent_draws:
            recent_draws = len(self.df)
            
        recent_data = self.df.tail(recent_draws)
        
        # Recent frequency analysis
        recent_main_nums = []
        recent_star_nums = []
        
        for col in ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']:
            recent_main_nums.extend(recent_data[col].tolist())
        for col in ['Star1', 'Star2']:
            recent_star_nums.extend(recent_data[col].tolist())
            
        self.recent_main_freq = Counter(recent_main_nums)
        self.recent_star_freq = Counter(recent_star_nums)
        
        # Hot and cold numbers
        self.hot_main_numbers = [num for num, freq in self.recent_main_freq.most_common(15)]
        self.cold_main_numbers = [num for num in range(1, 51) if self.recent_main_freq[num] <= 1]
        
        self.hot_star_numbers = [num for num, freq in self.recent_star_freq.most_common(6)]
        self.cold_star_numbers = [num for num in range(1, 13) if self.recent_star_freq[num] <= 1]
        
        print(f"Hot main numbers (frequent in last {recent_draws} draws): {self.hot_main_numbers}")
        print(f"Cold main numbers (rare in last {recent_draws} draws): {self.cold_main_numbers}")
        print(f"Hot star numbers: {self.hot_star_numbers}")
        print(f"Cold star numbers: {self.cold_star_numbers}")
        
    def predict_using_frequencies(self):
        """Predict using weighted frequency analysis."""
        print("\\nGenerating frequency-based predictions...")
        
        # Create weighted probabilities for main numbers
        total_main_freq = sum(self.main_frequencies.values())
        main_probabilities = {num: freq/total_main_freq for num, freq in self.main_frequencies.items()}
        
        # Adjust probabilities based on recent trends
        for num in self.hot_main_numbers[:10]:  # Boost hot numbers
            if num in main_probabilities:
                main_probabilities[num] *= 1.3
                
        for num in self.cold_main_numbers:  # Slightly boost cold numbers (due for appearance)
            if num in main_probabilities:
                main_probabilities[num] *= 1.1
                
        # Select main numbers using weighted random selection
        main_candidates = list(main_probabilities.keys())
        main_weights = [main_probabilities[num] for num in main_candidates]
        
        predicted_main = []
        remaining_candidates = main_candidates.copy()
        remaining_weights = main_weights.copy()
        
        for _ in range(5):
            if not remaining_candidates:
                break
                
            selected_idx = np.random.choice(len(remaining_candidates), p=np.array(remaining_weights)/sum(remaining_weights))
            selected_num = remaining_candidates[selected_idx]
            predicted_main.append(selected_num)
            
            # Remove selected number
            remaining_candidates.pop(selected_idx)
            remaining_weights.pop(selected_idx)
        
        # Similar process for star numbers
        total_star_freq = sum(self.star_frequencies.values())
        star_probabilities = {num: freq/total_star_freq for num, freq in self.star_frequencies.items()}
        
        for num in self.hot_star_numbers[:4]:
            if num in star_probabilities:
                star_probabilities[num] *= 1.3
                
        for num in self.cold_star_numbers:
            if num in star_probabilities:
                star_probabilities[num] *= 1.1
        
        star_candidates = list(star_probabilities.keys())
        star_weights = [star_probabilities[num] for num in star_candidates]
        
        predicted_stars = []
        remaining_star_candidates = star_candidates.copy()
        remaining_star_weights = star_weights.copy()
        
        for _ in range(2):
            if not remaining_star_candidates:
                break
                
            selected_idx = np.random.choice(len(remaining_star_candidates), p=np.array(remaining_star_weights)/sum(remaining_star_weights))
            selected_num = remaining_star_candidates[selected_idx]
            predicted_stars.append(selected_num)
            
            remaining_star_candidates.pop(selected_idx)
            remaining_star_weights.pop(selected_idx)
        
        return sorted(predicted_main), sorted(predicted_stars)
        
    def predict_using_patterns(self):
        """Predict using statistical patterns."""
        print("Generating pattern-based predictions...")
        
        # Target statistical properties based on historical data
        target_main_sum = int(self.df['Main_Sum'].mean() + np.random.normal(0, self.df['Main_Sum'].std() * 0.3))
        target_main_sum = max(15, min(225, target_main_sum))  # Reasonable bounds
        
        target_even_count = int(round(self.df['Main_Even_Count'].mean() + np.random.normal(0, 0.8)))
        target_even_count = max(0, min(5, target_even_count))
        
        # Generate candidates that match patterns
        best_candidates = []
        best_score = float('inf')
        
        for attempt in range(1000):  # Try many combinations
            # Generate random combination
            main_nums = sorted(np.random.choice(range(1, 51), 5, replace=False))
            star_nums = sorted(np.random.choice(range(1, 13), 2, replace=False))
            
            # Score based on how well it matches patterns
            main_sum = sum(main_nums)
            even_count = sum(1 for n in main_nums if n % 2 == 0)
            
            score = abs(main_sum - target_main_sum) + abs(even_count - target_even_count) * 10
            
            # Bonus for including frequent numbers
            freq_bonus = sum(self.main_frequencies.get(num, 0) for num in main_nums) / len(main_nums)
            score -= freq_bonus * 0.1
            
            # Bonus for balanced distribution
            if max(main_nums) - min(main_nums) > 20:  # Good spread
                score -= 5
                
            if score < best_score:
                best_score = score
                best_candidates = (main_nums, star_nums)
                
        return best_candidates
        
    def predict_using_gaps(self):
        """Predict using gap analysis (numbers due for appearance)."""
        print("Generating gap-based predictions...")
        
        # Calculate gaps since last appearance for each number
        main_gaps = {}
        star_gaps = {}
        
        # Initialize gaps
        for num in range(1, 51):
            main_gaps[num] = len(self.df)  # Maximum possible gap
        for num in range(1, 13):
            star_gaps[num] = len(self.df)
            
        # Find last appearance of each number
        for idx in range(len(self.df) - 1, -1, -1):  # Go backwards
            row = self.df.iloc[idx]
            
            # Check main numbers
            for col in ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']:
                num = row[col]
                gap = len(self.df) - 1 - idx
                if gap < main_gaps[num]:
                    main_gaps[num] = gap
                    
            # Check star numbers
            for col in ['Star1', 'Star2']:
                num = row[col]
                gap = len(self.df) - 1 - idx
                if gap < star_gaps[num]:
                    star_gaps[num] = gap
        
        # Select numbers with largest gaps (due for appearance)
        overdue_main = sorted(main_gaps.items(), key=lambda x: x[1], reverse=True)[:15]
        overdue_stars = sorted(star_gaps.items(), key=lambda x: x[1], reverse=True)[:6]
        
        # Combine with frequency analysis for balanced selection
        main_candidates = [num for num, gap in overdue_main]
        main_weights = [gap + self.main_frequencies.get(num, 0) for num, gap in overdue_main]
        
        star_candidates = [num for num, gap in overdue_stars]
        star_weights = [gap + self.star_frequencies.get(num, 0) for num, gap in overdue_stars]
        
        # Select using weighted probabilities
        predicted_main = sorted(np.random.choice(
            main_candidates, 5, replace=False, 
            p=np.array(main_weights)/sum(main_weights)
        ))
        
        predicted_stars = sorted(np.random.choice(
            star_candidates, 2, replace=False,
            p=np.array(star_weights)/sum(star_weights)  
        ))
        
        print(f"Numbers with largest gaps - Main: {[f'{num}({gap})' for num, gap in overdue_main[:10]]}")
        print(f"Numbers with largest gaps - Stars: {[f'{num}({gap})' for num, gap in overdue_stars[:6]]}")
        
        return predicted_main, predicted_stars
        
    def generate_ensemble_prediction(self):
        """Generate final prediction using ensemble of methods."""
        print("\\n" + "="*50)
        print("GENERATING ENSEMBLE PREDICTION")
        print("="*50)
        
        # Get predictions from different methods
        freq_main, freq_stars = self.predict_using_frequencies()
        pattern_main, pattern_stars = self.predict_using_patterns()
        gap_main, gap_stars = self.predict_using_gaps()
        
        print(f"\\nIndividual Method Predictions:")
        print(f"Frequency-based: Main {freq_main}, Stars {freq_stars}")
        print(f"Pattern-based:   Main {pattern_main}, Stars {pattern_stars}")
        print(f"Gap-based:       Main {gap_main}, Stars {gap_stars}")
        
        # Combine predictions (ensemble voting)
        main_votes = Counter()
        star_votes = Counter()
        
        for nums in [freq_main, pattern_main, gap_main]:
            for num in nums:
                main_votes[num] += 1
                
        for nums in [freq_stars, pattern_stars, gap_stars]:
            for num in nums:
                star_votes[num] += 1
        
        # Select most voted numbers
        final_main = [num for num, votes in main_votes.most_common(5)]
        final_stars = [num for num, votes in star_votes.most_common(2)]
        
        # If we don't have enough numbers, fill with frequency-based selection
        if len(final_main) < 5:
            remaining_main = [num for num in range(1, 51) if num not in final_main]
            remaining_main.sort(key=lambda x: self.main_frequencies.get(x, 0), reverse=True)
            final_main.extend(remaining_main[:5-len(final_main)])
            
        if len(final_stars) < 2:
            remaining_stars = [num for num in range(1, 13) if num not in final_stars]
            remaining_stars.sort(key=lambda x: self.star_frequencies.get(x, 0), reverse=True)
            final_stars.extend(remaining_stars[:2-len(final_stars)])
        
        return sorted(final_main), sorted(final_stars)
        
    def analyze_prediction_quality(self, main_nums, star_nums):
        """Analyze the quality of the prediction against historical patterns."""
        print(f"\\nPrediction Quality Analysis:")
        
        main_sum = sum(main_nums)
        star_sum = sum(star_nums)
        even_count = sum(1 for n in main_nums if n % 2 == 0)
        odd_count = 5 - even_count
        
        # Compare with historical averages
        avg_main_sum = self.df['Main_Sum'].mean()
        avg_star_sum = self.df['Star_Sum'].mean()
        avg_even = self.df['Main_Even_Count'].mean()
        
        print(f"Main sum: {main_sum} (historical avg: {avg_main_sum:.1f})")
        print(f"Star sum: {star_sum} (historical avg: {avg_star_sum:.1f})")
        print(f"Even/Odd: {even_count}/{odd_count} (historical avg: {avg_even:.1f}/{5-avg_even:.1f})")
        
        # Check frequency rankings
        main_freq_ranks = [sorted(self.main_frequencies.items(), key=lambda x: x[1], reverse=True).index((num, self.main_frequencies[num])) + 1 for num in main_nums]
        star_freq_ranks = [sorted(self.star_frequencies.items(), key=lambda x: x[1], reverse=True).index((num, self.star_frequencies[num])) + 1 for num in star_nums]
        
        print(f"Main number frequency rankings: {main_freq_ranks} (1=most frequent)")
        print(f"Star number frequency rankings: {star_freq_ranks}")
        
        print(f"Prediction includes {len(set(main_nums).intersection(set(self.hot_main_numbers)))} hot main numbers")
        print(f"Prediction includes {len(set(star_nums).intersection(set(self.hot_star_numbers)))} hot star numbers")


def main():
    """Main function to run the statistical prediction system."""
    print("Statistical EuroMillions Prediction System")
    print("=" * 50)
    
    # Set random seed for reproducibility (remove for random predictions)
    np.random.seed(42)
    
    try:
        # Initialize predictor
        predictor = StatisticalEuroMillionsPredictor()
        
        # Load and analyze data
        predictor.load_and_analyze_data()
        
        # Generate ensemble prediction
        predicted_main, predicted_stars = predictor.generate_ensemble_prediction()
        
        print(f"\\n" + "="*50)
        print("FINAL PREDICTION")
        print("="*50)
        print(f"Main Numbers: {predicted_main}")
        print(f"Star Numbers: {predicted_stars}")
        
        # Analyze prediction quality
        predictor.analyze_prediction_quality(predicted_main, predicted_stars)
        
        print(f"\\n" + "="*50)
        print("PREDICTION COMPLETE!")
        print("Good luck with your EuroMillions draw! 🍀")
        print("="*50)
        
        return {
            'main_numbers': predicted_main,
            'star_numbers': predicted_stars,
            'predictor': predictor
        }
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()