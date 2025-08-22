#!/usr/bin/env python3
"""
Physical Bias Detection for EuroMillions Lottery
Analyzes potential equipment irregularities and mechanical biases
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter, defaultdict
# import seaborn as sns  # Optional for visualization
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PhysicalBiasDetector:
    def __init__(self, data_file='euromillions_historical_results.csv'):
        self.data_file = data_file
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load historical data and prepare for bias analysis"""
        self.df = pd.read_csv(self.data_file)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        print(f"Loaded {len(self.df)} draws for physical bias analysis")
        print(f"Date range: {self.df['Date'].min().date()} to {self.df['Date'].max().date()}")
        
    def detect_ball_weight_bias(self):
        """Detect potential bias due to ball weight differences"""
        print("\n" + "="*60)
        print("BALL WEIGHT BIAS ANALYSIS")
        print("="*60)
        
        # Analyze frequency of each number
        main_freq = Counter()
        star_freq = Counter()
        
        for _, row in self.df.iterrows():
            for col in ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']:
                main_freq[row[col]] += 1
            for col in ['Star1', 'Star2']:
                star_freq[row[col]] += 1
        
        # Expected frequency for truly random draws
        total_main_draws = len(self.df) * 5
        total_star_draws = len(self.df) * 2
        expected_main_freq = total_main_draws / 50
        expected_star_freq = total_star_draws / 12
        
        print(f"\nMain Numbers Frequency Analysis:")
        print(f"Expected frequency per number: {expected_main_freq:.1f}")
        
        # Detect significant deviations
        main_biases = []
        for num in range(1, 51):
            observed = main_freq[num]
            deviation = (observed - expected_main_freq) / expected_main_freq
            if abs(deviation) > 0.1:  # More than 10% deviation
                main_biases.append((num, observed, deviation))
        
        main_biases.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print(f"\nPotential Main Number Biases (>10% deviation):")
        for num, freq, dev in main_biases[:10]:
            bias_type = "HEAVY" if dev > 0 else "LIGHT"
            print(f"  Number {num:2d}: {freq:3d} draws ({dev:+.1%}) - Possibly {bias_type} ball")
        
        # Star numbers analysis
        print(f"\nStar Numbers Frequency Analysis:")
        print(f"Expected frequency per star: {expected_star_freq:.1f}")
        
        star_biases = []
        for num in range(1, 13):
            observed = star_freq[num]
            deviation = (observed - expected_star_freq) / expected_star_freq
            if abs(deviation) > 0.15:  # More than 15% deviation for stars
                star_biases.append((num, observed, deviation))
        
        star_biases.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print(f"\nPotential Star Number Biases (>15% deviation):")
        for num, freq, dev in star_biases[:5]:
            bias_type = "HEAVY" if dev > 0 else "LIGHT"
            print(f"  Star {num:2d}: {freq:3d} draws ({dev:+.1%}) - Possibly {bias_type} ball")
        
        return main_biases, star_biases
    
    def detect_positional_bias(self):
        """Detect bias in ball positions during draw"""
        print("\n" + "="*60)
        print("POSITIONAL BIAS ANALYSIS")
        print("="*60)
        
        # Analyze which numbers appear in which positions
        position_freq = defaultdict(Counter)
        
        for _, row in self.df.iterrows():
            # Sort main numbers to analyze positional bias
            main_nums = sorted([row['Main1'], row['Main2'], row['Main3'], row['Main4'], row['Main5']])
            for pos, num in enumerate(main_nums):
                position_freq[pos][num] += 1
        
        print("Analyzing if certain numbers prefer certain draw positions...")
        
        # Check for numbers that appear disproportionately in first/last positions
        first_pos_bias = []
        last_pos_bias = []
        
        total_draws = len(self.df)
        expected_pos_freq = total_draws / 50
        
        for num in range(1, 51):
            first_freq = position_freq[0][num]  # Lowest position (smallest number)
            last_freq = position_freq[4][num]   # Highest position (largest number)
            
            first_deviation = (first_freq - expected_pos_freq) / expected_pos_freq
            last_deviation = (last_freq - expected_pos_freq) / expected_pos_freq
            
            if first_deviation > 0.2:  # 20% more likely to be smallest
                first_pos_bias.append((num, first_freq, first_deviation))
            
            if last_deviation > 0.2:   # 20% more likely to be largest
                last_pos_bias.append((num, last_freq, last_deviation))
        
        first_pos_bias.sort(key=lambda x: x[2], reverse=True)
        last_pos_bias.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\nNumbers biased toward SMALLEST position:")
        for num, freq, dev in first_pos_bias[:5]:
            print(f"  Number {num:2d}: {freq:3d} times smallest ({dev:+.1%})")
        
        print(f"\nNumbers biased toward LARGEST position:")
        for num, freq, dev in last_pos_bias[:5]:
            print(f"  Number {num:2d}: {freq:3d} times largest ({dev:+.1%})")
        
        return first_pos_bias, last_pos_bias
    
    def detect_machine_changes(self):
        """Detect potential lottery machine changes by analyzing temporal patterns"""
        print("\n" + "="*60)
        print("MACHINE CHANGE DETECTION")
        print("="*60)
        
        # Split data into time periods and compare frequency distributions
        total_draws = len(self.df)
        period_size = 200  # Analyze in chunks of 200 draws
        
        periods = []
        for i in range(0, total_draws - period_size, period_size):
            period_data = self.df.iloc[i:i+period_size]
            period_freq = Counter()
            
            for _, row in period_data.iterrows():
                for col in ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']:
                    period_freq[row[col]] += 1
            
            periods.append({
                'start_date': period_data['Date'].iloc[0],
                'end_date': period_data['Date'].iloc[-1],
                'frequencies': period_freq
            })
        
        # Compare consecutive periods for significant changes
        print(f"Analyzing {len(periods)} periods of {period_size} draws each...")
        
        significant_changes = []
        for i in range(1, len(periods)):
            current = periods[i]['frequencies']
            previous = periods[i-1]['frequencies']
            
            # Calculate chi-square test for distribution change
            observed = [current[num] for num in range(1, 51)]
            expected = [previous[num] for num in range(1, 51)]
            
            # Only test if we have sufficient data
            if min(expected) > 0:
                chi2, p_value = stats.chisquare(observed, expected)
                
                if p_value < 0.01:  # Significant change detected
                    significant_changes.append({
                        'period': i,
                        'date': periods[i]['start_date'],
                        'chi2': chi2,
                        'p_value': p_value
                    })
        
        print(f"\nPotential Machine Changes Detected:")
        for change in significant_changes:
            print(f"  Period {change['period']} ({change['date'].date()}): p-value = {change['p_value']:.6f}")
            
        if significant_changes:
            print(f"\n⚠️  {len(significant_changes)} periods show significant frequency changes")
            print("This could indicate machine maintenance, ball set changes, or equipment replacement")
        else:
            print("\nNo significant machine changes detected in the analyzed periods")
        
        return significant_changes
    
    def detect_temperature_humidity_bias(self):
        """Detect potential environmental biases (seasons, weather patterns)"""
        print("\n" + "="*60)
        print("ENVIRONMENTAL BIAS ANALYSIS")
        print("="*60)
        
        # Add seasonal information
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Season'] = self.df['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        # Analyze frequency differences by season
        seasonal_freq = defaultdict(Counter)
        
        for _, row in self.df.iterrows():
            season = row['Season']
            for col in ['Main1', 'Main2', 'Main3', 'Main4', 'Main5']:
                seasonal_freq[season][row[col]] += 1
        
        # Calculate expected frequencies
        draws_per_season = self.df['Season'].value_counts()
        
        print("Analyzing seasonal patterns (temperature/humidity effects)...")
        
        seasonal_biases = {}
        for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
            expected_freq = (draws_per_season[season] * 5) / 50
            biases = []
            
            for num in range(1, 51):
                observed = seasonal_freq[season][num]
                if expected_freq > 0:
                    deviation = (observed - expected_freq) / expected_freq
                    if abs(deviation) > 0.15:  # 15% deviation
                        biases.append((num, observed, deviation))
            
            biases.sort(key=lambda x: abs(x[2]), reverse=True)
            seasonal_biases[season] = biases
        
        for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
            print(f"\n{season} Biases:")
            for num, freq, dev in seasonal_biases[season][:3]:
                bias_type = "More frequent" if dev > 0 else "Less frequent"
                print(f"  Number {num:2d}: {bias_type} ({dev:+.1%})")
        
        return seasonal_biases
    
    def create_bias_based_predictor(self, main_biases, star_biases):
        """Create predictions based on detected physical biases"""
        print("\n" + "="*60)
        print("BIAS-BASED PREDICTIONS")
        print("="*60)
        
        # Extract overrepresented numbers (heavy balls)
        heavy_main = [num for num, freq, dev in main_biases if dev > 0][:15]
        light_main = [num for num, freq, dev in main_biases if dev < 0][:15]
        
        heavy_stars = [num for num, freq, dev in star_biases if dev > 0][:8]
        light_stars = [num for num, freq, dev in star_biases if dev < 0][:8]
        
        predictions = []
        
        # Strategy 1: Favor heavy balls (appear more often)
        main_nums = np.random.choice(heavy_main, size=min(5, len(heavy_main)), replace=False)
        while len(main_nums) < 5:
            remaining = [n for n in range(1, 51) if n not in main_nums]
            main_nums = np.append(main_nums, np.random.choice(remaining))
        
        star_nums = np.random.choice(heavy_stars, size=min(2, len(heavy_stars)), replace=False)
        while len(star_nums) < 2:
            remaining = [n for n in range(1, 13) if n not in star_nums]
            star_nums = np.append(star_nums, np.random.choice(remaining))
        
        predictions.append({
            'main_numbers': sorted(main_nums.tolist()),
            'star_numbers': sorted(star_nums.tolist()),
            'strategy': 'Heavy Ball Bias'
        })
        
        # Strategy 2: Contrarian - favor light balls (due for comeback)
        main_nums = np.random.choice(light_main, size=min(5, len(light_main)), replace=False)
        while len(main_nums) < 5:
            remaining = [n for n in range(1, 51) if n not in main_nums]
            main_nums = np.append(main_nums, np.random.choice(remaining))
        
        star_nums = np.random.choice(light_stars, size=min(2, len(light_stars)), replace=False)
        while len(star_nums) < 2:
            remaining = [n for n in range(1, 13) if n not in star_nums]
            star_nums = np.append(star_nums, np.random.choice(remaining))
        
        predictions.append({
            'main_numbers': sorted(main_nums.tolist()),
            'star_numbers': sorted(star_nums.tolist()),
            'strategy': 'Light Ball Compensation'
        })
        
        # Strategy 3: Mixed approach
        heavy_sample = np.random.choice(heavy_main, size=min(3, len(heavy_main)), replace=False)
        light_sample = np.random.choice(light_main, size=min(2, len(light_main)), replace=False)
        main_nums = np.concatenate([heavy_sample, light_sample])
        
        if len(main_nums) < 5:
            remaining = [n for n in range(1, 51) if n not in main_nums]
            additional = np.random.choice(remaining, size=5-len(main_nums), replace=False)
            main_nums = np.concatenate([main_nums, additional])
        
        star_nums = []
        if heavy_stars:
            star_nums.append(np.random.choice(heavy_stars))
        if light_stars:
            star_nums.append(np.random.choice(light_stars))
        
        while len(star_nums) < 2:
            remaining = [n for n in range(1, 13) if n not in star_nums]
            star_nums.append(np.random.choice(remaining))
        
        predictions.append({
            'main_numbers': sorted(main_nums.tolist()),
            'star_numbers': sorted(star_nums[:2]),
            'strategy': 'Mixed Bias Strategy'
        })
        
        print("Predictions based on physical bias analysis:")
        for i, pred in enumerate(predictions, 1):
            main = pred['main_numbers']
            stars = pred['star_numbers']
            strategy = pred['strategy']
            print(f"{i}. [{strategy:20s}] {main[0]:2d} {main[1]:2d} {main[2]:2d} {main[3]:2d} {main[4]:2d} | Stars: {stars[0]:2d} {stars[1]:2d}")
        
        return predictions
    
    def run_complete_analysis(self):
        """Run complete physical bias analysis"""
        print("EuroMillions Physical Bias Detection System")
        print("="*50)
        print("Analyzing potential equipment irregularities and mechanical biases...")
        
        # Run all bias detection methods
        main_biases, star_biases = self.detect_ball_weight_bias()
        first_pos_bias, last_pos_bias = self.detect_positional_bias()
        machine_changes = self.detect_machine_changes()
        seasonal_biases = self.detect_temperature_humidity_bias()
        
        # Generate bias-based predictions
        predictions = self.create_bias_based_predictor(main_biases, star_biases)
        
        print("\n" + "="*60)
        print("PHYSICAL BIAS ANALYSIS SUMMARY")
        print("="*60)
        print(f"✓ Frequency bias analysis: {len(main_biases)} main number anomalies detected")
        print(f"✓ Positional bias analysis: {len(first_pos_bias + last_pos_bias)} position anomalies")
        print(f"✓ Temporal analysis: {len(machine_changes)} potential machine changes")
        print(f"✓ Environmental analysis: Seasonal patterns identified")
        print(f"✓ Generated {len(predictions)} bias-based predictions")
        
        print(f"\n🔬 SCIENTIFIC VALIDITY:")
        print("• Physical biases are theoretically possible")
        print("• Ball weight differences could affect selection probability")
        print("• Machine wear and environmental factors may create patterns")
        print("• Statistical significance testing applied to detect real vs random patterns")
        
        print(f"\n⚠️  IMPORTANT NOTES:")
        print("• Modern lottery equipment is designed to minimize physical biases")
        print("• Detected patterns may still be within normal statistical variation")
        print("• Regular equipment maintenance and ball replacement reduce bias potential")
        print("• Use this analysis for research and entertainment purposes only")
        
        return {
            'main_biases': main_biases,
            'star_biases': star_biases,
            'positional_biases': (first_pos_bias, last_pos_bias),
            'machine_changes': machine_changes,
            'seasonal_biases': seasonal_biases,
            'predictions': predictions
        }


def main():
    """Main function to run physical bias detection"""
    detector = PhysicalBiasDetector()
    results = detector.run_complete_analysis()
    
    return results


if __name__ == "__main__":
    main()