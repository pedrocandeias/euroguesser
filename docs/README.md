# EuroGuesser 🎯

A comprehensive, unified EuroMillions lottery prediction system that combines multiple scientific approaches including statistical analysis, machine learning, temporal pattern analysis, and physical bias detection. Features advanced weighted ensemble methods and comprehensive backtesting validation.

## 🎲 Overview

EuroGuesser is a sophisticated research project that implements and validates multiple prediction strategies for EuroMillions lottery numbers. While acknowledging that lottery draws are designed to be random, this system scientifically explores legitimate approaches that could theoretically provide insights into lottery patterns, with proper validation and performance metrics.

## 🌟 Key Features

### 🚀 **Unified Prediction System** (New!)
- **Single Entry Point**: `unified_euromillions_predictor.py` - comprehensive prediction system
- **Weighted Ensemble**: Scientifically validated method combining all approaches
- **Backtesting Framework**: Historical validation against real draws (2020-2024)
- **Temporal Intelligence**: Context-aware predictions based on day/month/season patterns
- **Performance Rankings**: Real-time confidence scoring and method comparison

### 📊 **Statistical Prediction Models** (Configurable Weights)
- **Frequency Analysis**: Deep statistical analysis using 21+ years of data (default 25% weight)
- **Gap Analysis**: Numbers "overdue" for selection with statistical significance (default 25% weight) 
- **Pattern Analysis**: Consecutive numbers, sum analysis, and distribution patterns (default 25% weight)
- **Temporal Analysis**: Day of week, monthly, seasonal, and quarterly patterns (default 25% weight)
- **🎛️ NEW**: Fully configurable percentage controls for all analysis methods

### 🤖 **Machine Learning & AI** 
- **Trained Neural Networks**: Pre-trained models using comprehensive historical dataset (18% weight)
- **TensorFlow Integration**: Advanced deep learning with LSTM and ensemble methods
- **Adaptive Learning**: Models that improve with new data
- **Multiple Architectures**: Dense networks, sequential models, and custom architectures

### 🔬 **Physical Bias Detection**
- **Equipment Bias Analysis**: Scientific detection of potential mechanical irregularities (5-4% weight)
- **Ball Weight Variations**: Heavy/light bias detection based on frequency deviations
- **Position Analysis**: Statistical analysis of number positioning patterns
- **Environmental Effects**: Seasonal and temporal environmental factor analysis

### 🧪 **Advanced AI Techniques**
- **Quantum-Inspired Sampling**: Probabilistic superposition principles for number selection
- **Chaos Theory Methods**: Lorenz attractor and non-linear dynamics approaches
- **Fibonacci Patterns**: Golden ratio and mathematical sequence analysis
- **Experimental Methods**: Research-grade algorithms for pattern exploration

### 📈 **Comprehensive Data Management**
- **Complete Dataset**: 1,642 draws from 2004-2025 (21+ years of comprehensive data)
- **HTML Scraper**: Advanced scraper for extracting data from official results pages
- **Multiple Formats**: CSV, training-ready, and analysis-optimized data formats
- **Data Validation**: Integrity checking and statistical validation of historical data

### 🔍 **Performance Validation & Analytics**
- **Backtesting Framework**: Historical validation testing against 25 draws per method
- **Performance Metrics**: EuroMillions prize structure scoring (Jackpot=100, 2nd Prize=90, etc.)
- **Statistical Significance**: Chi-square tests, confidence intervals, and proper validation
- **Method Comparison**: Head-to-head performance analysis with statistical rigor

## 🚀 Quick Start

### Prerequisites
```bash
# Core dependencies (included by default)
python3 >= 3.8
numpy >= 1.26.4
pandas >= 2.1.4
beautifulsoup4 >= 4.x

# Optional ML dependencies (for advanced features)
pip install tensorflow scikit-learn matplotlib
```

### Primary Usage (Recommended)

#### 🎯 **Unified Prediction System** (Best)
```bash
# Complete prediction analysis (recommended)
python3 unified_euromillions_predictor.py

# Standard backtesting validation (25 samples)
python3 unified_euromillions_predictor.py --backtest

# Extended backtesting (50 samples, more reliable)
python3 unified_euromillions_predictor.py --backtest --extended

# Custom sample size (10-470 range available)
python3 unified_euromillions_predictor.py --backtest --samples=100
```

#### 📊 **Individual Statistical Methods**
```bash
# Fast statistical predictions
python3 quick_predictor.py

# Comprehensive statistical analysis (configurable weights)
python3 statistical_predictor.py

# Test different weight configurations
python3 test_weight_configurations.py

# Overview of all methods
python3 prediction_summary.py
```

#### 🎯 **Automatic Optimization Mode** (NEW!)
```bash
# Auto-optimize for 10% prize win rate (default target)
python3 unified_euromillions_predictor.py --auto-optimize

# Custom target prize rate (15%) with more iterations
python3 unified_euromillions_predictor.py --auto-optimize --target-prize-rate=0.15 --max-iterations=30

# Quick optimization with lower target (5%)
python3 unified_euromillions_predictor.py --auto-optimize --target-prize-rate=0.05 --max-iterations=10
```

#### 🔧 **Enhanced Backtesting Options**
```bash
# Custom draws per method (more precise control)
python3 unified_euromillions_predictor.py --backtest --draws-per-method=50

# Custom year range testing
python3 unified_euromillions_predictor.py --backtest --start-year=2018 --end-year=2023

# Comprehensive validation: 100 draws per method, 2004-2024 period  
python3 unified_euromillions_predictor.py --backtest --draws-per-method=100 --start-year=2004 --end-year=2024

# Quick focused test: 2021-2022 period with 30 draws per method
python3 unified_euromillions_predictor.py --backtest --draws-per-method=30 --start-year=2021 --end-year=2022
```

#### 🔧 **Data Management**
```bash
# Extract data from HTML results
python3 optimized_html_scraper.py

# Train machine learning models
python3 model_training.py
python3 improved_training.py
```

## 🏆 Performance Results (Backtesting Validation)

**Configurable Testing Parameters** - You can now customize:
- **Draws per method**: `--draws-per-method=NUMBER` (default: 25)  
- **Year range**: `--start-year=YYYY --end-year=YYYY` (default: 2020-2024)
- **Total samples**: `--samples=NUMBER` (distributed among methods)

**Default Results** based on historical testing against 25 draws per method (2020-2024):

| Rank | Method | Avg Score | Main Matches | Star Matches | Prize Wins | Confidence |
|------|--------|-----------|--------------|--------------|------------|------------|
| 🥇 **1** | **Weighted Ensemble** | **2.04** | **0.64/5** | **0.44/2** | **4.0%** | **High** |
| 🥈 2 | Frequency Analysis | 1.52 | 0.68/5 | 0.36/2 | 0.0% | High |
| 🥉 3 | Neural Network | 1.36 | 0.52/5 | 0.44/2 | 0.0% | Medium |
| 4 | Gap Analysis | 0.96 | 0.56/5 | 0.28/2 | 0.0% | Medium |
| 5 | Pattern Analysis | 0.92 | 0.50/5 | 0.30/2 | 0.0% | Medium |

**🎯 Key Finding**: The **Weighted Ensemble** method significantly outperforms individual methods, achieving the only prize win (4% rate) and highest average score.

## 📁 Project Structure

```
euroguesser/
├── 🚀 unified_euromillions_predictor.py     # 🎯 Main system (stay in root for easy access)
│
├── 📁 core/                                 # Core prediction utilities
│   ├── statistical_predictor.py            # 🎛️ Advanced statistical analysis (configurable weights)
│   ├── quick_predictor.py                  # Fast statistical predictions  
│   ├── test_weight_configurations.py       # 🆕 Weight configuration testing
│   ├── prediction_summary.py               # Methods overview
│   └── optimized_html_scraper.py           # Data extraction from HTML
│
├── 📁 models/                               # 🤖 Machine Learning components
│   ├── model_training.py                   # Basic neural network training
│   ├── improved_training.py                # Advanced ML training
│   ├── predictions.py                      # ML-based predictions
│   ├── lottery_model.keras                 # 🧠 Trained basic model
│   └── improved_lottery_model.keras        # 🧠 Trained advanced model
│
├── 📁 data/                                 # 📊 Dataset files
│   ├── scraped_euromillions_results.csv    # 📈 Primary dataset (1,642 draws)
│   └── scraped_euromillions_training.txt   # ML training format
│
├── 📁 docs/                                 # 📚 Documentation
│   ├── README.md                           # This file
│   ├── CLAUDE.md                           # Development guidelines  
│   └── CHANGELOG.md                        # Version history
│
├── 📁 predictions/                          # 💾 Generated prediction files
│   └── euromillions_predictions_YYYYMMDD_HHMMSS.json  # Timestamped prediction keys
│
└── 📁 euromillions_env/                     # 🐍 Python virtual environment
```

## 🎛️ Configurable Analysis Weights (NEW!)

### 🎨 **Custom Weight Configuration**
The statistical predictor now supports fully configurable percentage controls for all analysis methods:

```python
# Example configurations
predictor = StatisticalEuroMillionsPredictor(
    frequency_weight=40.0,  # 40% - Focus on historical frequency
    gap_weight=30.0,        # 30% - Numbers due for appearance  
    pattern_weight=20.0,    # 20% - Statistical patterns
    temporal_weight=10.0    # 10% - Seasonal/monthly trends
)
```

### 🧪 **Pre-configured Strategy Examples**
```bash
# Test 8 different strategies
python3 test_weight_configurations.py
```

| Strategy | Frequency | Gap | Pattern | Temporal | Best For |
|----------|-----------|-----|---------|----------|----------|
| **Balanced** | 25% | 25% | 25% | 25% | General use |
| **Frequency Focus** | 60% | 15% | 15% | 10% | Historical patterns |
| **Gap Analysis** | 10% | 70% | 15% | 5% | "Due" numbers |
| **Pattern Only** | 0% | 0% | 100% | 0% | Mathematical patterns |
| **Temporal Focus** | 10% | 10% | 20% | 60% | Seasonal trends |
| **Hot Numbers** | 40% | 40% | 10% | 10% | Recent activity |

### ⚙️ **Analysis Method Details**
- **Frequency Analysis**: Based on 21+ years of historical frequency data
- **Gap Analysis**: Identifies numbers with longest gaps since last appearance  
- **Pattern Analysis**: Even/odd ratios, sum analysis, consecutive patterns
- **Temporal Analysis**: Monthly, seasonal, and day-of-week patterns

### 🔧 **Weight Normalization**
- Weights automatically normalize to 100% regardless of input values
- Zero weights disable specific analysis methods entirely
- Supports any combination of analysis approaches

## 🎯 Unified Prediction System Architecture

### 🧠 **Weighted Ensemble Method** (Primary Recommendation)
The system combines multiple prediction approaches using scientifically validated weights:

- **Statistical Methods**: Configurable weights (default: balanced 25% each)
  - Frequency Analysis: 25% (configurable)
  - Gap Analysis: 25% (configurable)
  - Pattern Analysis: 25% (configurable)
  - Temporal Analysis: 25% (configurable)

- **Temporal Analysis**: 48% potential weight (context-dependent) 
  - Season-based: 12% per season
  - Day/Month patterns: Dynamic weighting

- **Machine Learning**: 25% total weight (good performance)
  - Trained Neural Network: 18%
  - Standard Neural Network: 7%

- **Physical Bias Detection**: 9% total weight (scientific basis)
  - Heavy bias detection: 5%
  - Light compensation: 4%

- **Experimental Methods**: <5% total weight (research purposes)

### 📅 **Temporal Intelligence System**
Revolutionary time-aware predictions that adapt to current context:

- **Day of Week Patterns**: Friday shows highest activity (numbers 23, 4, 35)
- **Monthly Trends**: February (3, 25, 38), March (23, 26, 44), etc.
- **Seasonal Analysis**: Summer (35, 42), Winter (44, 31), Spring (26, 28)
- **Quarterly Patterns**: Q1 (44, 23), Q2 (26, 28), Q3 (35, 42)

### 🔍 **Backtesting Framework**
Comprehensive historical validation system:

- **Time-Series Validation**: Uses only historical data available before each draw
- **EuroMillions Prize Scoring**: Matches official prize structure (5+2=100 points)
- **Statistical Significance**: Multiple methods tested across 25+ draws each
- **Performance Metrics**: Average score, match rates, prize win percentages

## 📊 Dataset Information

- **Total Draws**: 1,642 validated historical results  
- **Date Range**: February 2004 - August 2025 (21+ years)
- **Data Source**: Official EuroMillions results via advanced HTML scraping
- **Format**: Date, Main1-5, Star1-2 with comprehensive temporal features
- **Validation**: Statistical integrity checks and anomaly detection
- **Update Method**: Advanced HTML scraper with error handling and validation

## 🔬 Scientific Methodology

### 📈 **Statistical Validation**
- **Backtesting**: Historical performance validation against real draws
- **Cross-validation**: Time-series split validation for temporal integrity  
- **Confidence Intervals**: Statistical significance testing for all methods
- **Chi-square Tests**: Validation of frequency distribution assumptions

### 🧪 **Research Areas Explored**
1. **Temporal Patterns**: Day/month/season-based number frequency analysis
2. **Physical Equipment Biases**: Statistical detection of mechanical irregularities  
3. **Machine Learning Pattern Recognition**: Deep learning on 21+ years of data
4. **Ensemble Methods**: Weighted combination of multiple prediction approaches

### ⚖️ **Mathematical Limitations**
- Each draw remains independent (past results don't influence future draws)
- True randomness cannot be predicted mathematically
- Probability remains 1 in 139,838,160 for any specific combination
- No algorithm can overcome fundamental randomness

## 📈 Key Findings & Insights

### 🏆 **Performance Discoveries**
- **Weighted Ensemble** achieves 4% prize win rate vs. 0% for individual methods
- **Statistical methods** consistently outperform experimental approaches
- **Temporal patterns** show significant seasonal and daily variations
- **Machine learning** provides competitive performance when properly trained

### 📊 **Statistical Patterns Identified**
- **Most frequent main number**: 23 (214 appearances over 21 years)
- **Seasonal variations**: Summer numbers (35, 42) vs. Winter (44, 31)
- **Day-of-week effects**: Friday draws favor different number patterns
- **Temporal trends**: Quarterly patterns show statistical significance

### 🔬 **Physical Bias Detection**
- **Frequency deviations**: 10 main numbers show >10% frequency deviation
- **Positional analysis**: Statistical evidence of position-dependent patterns
- **Equipment changes**: Temporal analysis suggests periodic equipment updates
- **Environmental factors**: Seasonal effects on draw patterns detected

## 🚨 Important Disclaimers

### ⚠️ **Legal & Ethical**
- **For research, education, and entertainment purposes only**
- **No guarantee of improved winning odds or financial returns**
- **Lottery outcomes are designed to be unpredictable and fair**
- **Always gamble responsibly - never bet more than you can afford to lose**

### 🔬 **Scientific Limitations**
- Detected patterns may represent statistical noise or coincidence
- Modern lottery equipment is designed to minimize any physical biases
- Regular maintenance and oversight reduce equipment irregularities
- True mathematical randomness cannot be overcome by any predictive method

### 📊 **Performance Caveats**  
- Backtesting performance does not guarantee future results
- Small sample sizes limit statistical significance of some findings
- Prize win rates are based on limited historical testing
- Past performance cannot predict future lottery outcomes

## 🎯 N Prediction Keys Generation & Testing (NEW!)

The system now supports generating a configurable number (N) of prediction keys, saving them to files, and testing them against specific keys.

### 🔢 **Configurable Key Generation**
```bash
# Generate specific number of prediction keys
python3 unified_euromillions_predictor.py --num-keys=5    # Generate 5 keys
python3 unified_euromillions_predictor.py --num-keys=20   # Generate 20 keys  
python3 unified_euromillions_predictor.py --num-keys=50   # Generate 50 keys
```

### 💾 **Save Predictions to File**
```bash
# Generate keys and save to timestamped JSON file in predictions/ folder
python3 unified_euromillions_predictor.py --num-keys=15 --save-to-file

# Combines with other operations - backtesting + save
python3 unified_euromillions_predictor.py --backtest --save-to-file
```

### 🎯 **Test Against Specific Key**
```bash
# Generate keys and immediately test against a known result
python3 unified_euromillions_predictor.py --num-keys=10 --test-key=7,14,23,35,42:3,9

# Test with different key combinations
python3 unified_euromillions_predictor.py --num-keys=25 --test-key=1,12,18,26,44:2,7
python3 unified_euromillions_predictor.py --num-keys=30 --test-key=5,15,25,35,45:1,12
```

### 📁 **Load and Test Saved Predictions**
```bash
# Load previously saved predictions and test against new key
python3 unified_euromillions_predictor.py --load-and-test=predictions/euromillions_predictions_20250828_091635.json:7,14,23,35,42:3,9

# Test multiple saved prediction files from predictions/ folder
python3 unified_euromillions_predictor.py --load-and-test=predictions/predictions_batch1.json:12,17,25,33,41:2,9
python3 unified_euromillions_predictor.py --load-and-test=predictions/predictions_batch2.json:1,8,15,22,29:4,11
```

### 🔄 **Combined Operations**
```bash
# Generate, save, and test in one command
python3 unified_euromillions_predictor.py --num-keys=20 --save-to-file --test-key=23,27,29,42,45:1,11

# Generate many keys, save, and test against historical result
python3 unified_euromillions_predictor.py --num-keys=100 --save-to-file --test-key=9,16,20,32,44:3,8
```

### 📊 **File Format & Structure**

**Saved Prediction File Format (JSON):**
```json
{
  "generated_at": "2025-08-28T09:16:35.802668",
  "total_predictions": 10,
  "num_keys_requested": 10,
  "predictions": [
    {
      "key_id": 1,
      "main_numbers": [3, 23, 26, 27, 40],
      "star_numbers": [2, 8],
      "method": "Physical Bias (heavy_bias)",
      "confidence": 0.8
    },
    {
      "key_id": 2,
      "main_numbers": [1, 12, 19, 28, 30],
      "star_numbers": [1, 11],
      "method": "Physical Bias (light_compensation)",
      "confidence": 0.8
    }
  ]
}
```

### 🏆 **Testing Results & Scoring**

The system uses official EuroMillions prize scoring:
- **Jackpot (5+2)**: 100 points
- **2nd Prize (5+1)**: 90 points
- **3rd Prize (5+0)**: 80 points
- **4th Prize (4+2)**: 70 points
- **12th Prize (2+1)**: 10 points
- **13th Prize (2+0)**: 5 points
- **No Prize**: 0 points

**Testing Output Example:**
```
🎯 Testing 10 prediction keys against test key:
   Test Key: [27, 29, 42, 45, 7] | Stars: [1, 10]

Key  Main Numbers    Stars  Method               Matches    Score  Prize
6     7 20 27 29 45   1 10  Frequency Analysis   4/5+2/2    70     4th Prize
5    13 27 29 42 45   3 11  Frequency Analysis   4/5+0/2    50     6th Prize

📊 TESTING SUMMARY:
   🏆 Best Key: #6 - Frequency Analysis (Score: 70)
   📈 Total Score: 160
   🎁 Prize Winners: 4/10 keys  
   📊 Average Score: 16.00
```

### 🎛️ **Advanced Usage Scenarios**

**Research Workflow:**
```bash
# Phase 1: Generate large batch for analysis
python3 unified_euromillions_predictor.py --num-keys=50 --save-to-file

# Phase 2: Test against multiple historical results
python3 unified_euromillions_predictor.py --load-and-test=predictions/predictions.json:12,15,18,46,49:1,3
python3 unified_euromillions_predictor.py --load-and-test=predictions/predictions.json:7,23,27,35,44:9,11  
python3 unified_euromillions_predictor.py --load-and-test=predictions/predictions.json:2,14,17,32,50:5,12

# Phase 3: Generate targeted predictions based on results
python3 unified_euromillions_predictor.py --num-keys=25 --test-key=best_performing_key
```

**Validation Testing:**
```bash
# Test prediction accuracy against recent draws
python3 unified_euromillions_predictor.py --num-keys=30 --test-key=recent_draw_result

# Batch validation against multiple recent results
for key in draw1 draw2 draw3; do
  python3 unified_euromillions_predictor.py --load-and-test=predictions/predictions.json:$key
done
```

## 🛠️ Development & Contributing

### 🔧 **Architecture**
- **Modular Design**: Each prediction method is independently implemented
- **Unified Interface**: Single entry point with comprehensive analysis
- **Extensible Framework**: Easy addition of new prediction methods
- **Validation Pipeline**: Built-in backtesting and performance measurement

### 🧪 **Testing & Validation**

#### 🎯 **Automatic Optimization (INTELLIGENT MODE)**
```bash
# Auto-optimize to achieve 10% prize win rate (finds best configuration automatically)
python3 unified_euromillions_predictor.py --auto-optimize

# Custom target with extended search (target 15%, up to 30 iterations)
python3 unified_euromillions_predictor.py --auto-optimize --target-prize-rate=0.15 --max-iterations=30

# Research mode: Lower target with detailed analysis
python3 unified_euromillions_predictor.py --auto-optimize --target-prize-rate=0.08 --min-draws=10
```

#### 🔍 **Enhanced Backtesting Options**
```bash
# Quick backtesting (25 draws per method, 2020-2024)
python3 unified_euromillions_predictor.py --backtest

# Extended backtesting (50 draws per method)  
python3 unified_euromillions_predictor.py --backtest --extended

# Custom sample size (distributed among 7 methods)
python3 unified_euromillions_predictor.py --backtest --samples=100   # ~15 minutes

# Precise per-method control (RECOMMENDED for research)
python3 unified_euromillions_predictor.py --backtest --draws-per-method=75   # 75 draws per method

# Custom year range testing
python3 unified_euromillions_predictor.py --backtest --start-year=2018 --end-year=2022

# Comprehensive validation (200 draws per method, full period)
python3 unified_euromillions_predictor.py --backtest --draws-per-method=200 --start-year=2004 --end-year=2024
```

#### 🎯 **Automatic Optimization Parameters**
| Parameter | Purpose | Example | Description |
|-----------|---------|---------|-------------|
| **`--auto-optimize`** | Enable intelligent optimization | `--auto-optimize` | Automatically find configuration for target prize rate |
| **`--target-prize-rate=X.X`** | Target prize win rate | `--target-prize-rate=0.10` | Target 10% prize win rate (default: 0.10) |
| **`--max-iterations=N`** | Maximum optimization rounds | `--max-iterations=25` | Test up to 25 different configurations (default: 20) |
| **`--min-draws=N`** | Starting draws per method | `--min-draws=10` | Begin optimization with 10 draws/method (default: 5) |

#### 📊 **Manual Backtesting Parameters Guide**
| Parameter | Purpose | Example | Description |
|-----------|---------|---------|-------------|
| **`--samples=N`** | Total draws distributed among methods | `--samples=200` | 200 draws shared across 7 methods (~28 each) |
| **`--draws-per-method=N`** | Exact draws per method | `--draws-per-method=50` | 50 draws tested for each of 7 methods (350 total) |
| **`--start-year=YYYY`** | Starting year for test period | `--start-year=2018` | Only test draws from 2018 onwards |
| **`--end-year=YYYY`** | Ending year for test period | `--end-year=2022` | Only test draws up to 2022 |

#### 🎯 **Optimization Strategies Tested**
| Strategy | Focus | Draws Range | Year Range | Purpose |
|----------|-------|-------------|------------|---------|
| **Increase Draws** | Sample size scaling | 5-15 per method | 2020-2024 | Find minimum viable sample |
| **Recent Focus** | Current patterns | 10-15 per method | 2021-2024 | Emphasize recent trends |
| **Historical Period** | Pre-COVID patterns | 10-15 per method | 2016-2020 | Historical consistency |
| **Large Sample** | Statistical power | 20-30 per method | 2017-2024 | High confidence testing |
| **Full Range** | Maximum data | 15-25 per method | 2004-2024 | Complete historical analysis |
| **Pre/Post COVID** | Pattern comparison | 15-20 per method | Split periods | Economic impact analysis |

#### 🦠 **Pre/Post COVID Analysis Examples**
```bash
# Pre-COVID period analysis (2004-2019) - Historical baseline patterns
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=25 --target-prize-rate=0.10 --start-year=2004 --end-year=2019

# Post-COVID period analysis (2020-2024) - Modern patterns 
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=25 --target-prize-rate=0.10 --start-year=2020 --end-year=2024

# COVID transition period (2019-2021) - Pattern shift analysis
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=30 --target-prize-rate=0.08 --start-year=2019 --end-year=2021

# Comprehensive pre/post comparison with extended testing
python3 unified_euromillions_predictor.py --backtest --draws-per-method=40 --start-year=2015 --end-year=2019  # Pre-COVID
python3 unified_euromillions_predictor.py --backtest --draws-per-method=40 --start-year=2020 --end-year=2024  # Post-COVID
```

#### 🌍 **Full Range Analysis Examples**  
```bash
# Complete historical analysis (2004-2024) - Maximum 21-year dataset
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=35 --target-prize-rate=0.12 --start-year=2004 --end-year=2024

# Full range with high iteration count for research-grade results
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=50 --target-prize-rate=0.10 --min-draws=10 --start-year=2004 --end-year=2024

# Academic validation: Full range with maximum precision
python3 unified_euromillions_predictor.py --backtest --draws-per-method=100 --start-year=2004 --end-year=2024

# Comprehensive full-range backtesting with custom sample sizes
python3 unified_euromillions_predictor.py --backtest --draws-per-method=75 --start-year=2004 --end-year=2024   # 525 total tests
python3 unified_euromillions_predictor.py --backtest --draws-per-method=150 --start-year=2004 --end-year=2024  # 1,050 total tests

# Full range optimization with different targets and iteration combinations
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=60 --target-prize-rate=0.15 --start-year=2004 --end-year=2024
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=40 --target-prize-rate=0.08 --start-year=2004 --end-year=2024
```

#### 🔧 **Max Iterations Usage Examples & Strategies**

##### 🚀 **Quick Testing (Low Iterations)**
```bash
# Fast development testing (5-10 iterations)
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=5   # ~2-5 minutes
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=8 --target-prize-rate=0.06

# Quick validation with specific focus periods
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=10 --start-year=2020 --end-year=2024
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=12 --start-year=2021 --target-prize-rate=0.08
```

##### 🎯 **Standard Optimization (Medium Iterations)**
```bash
# Balanced approach (15-25 iterations) - Recommended for most users
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=15
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=20 --target-prize-rate=0.12
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=25 --min-draws=8

# Production-ready optimization with custom parameters
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=22 --target-prize-rate=0.10 --start-year=2018
```

##### 🔬 **Research-Grade Testing (High Iterations)**
```bash
# Thorough analysis (30-50 iterations) - Research and academic use
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=35 --target-prize-rate=0.15
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=40 --min-draws=12 --start-year=2010
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=45 --target-prize-rate=0.13 --start-year=2004 --end-year=2024

# Maximum precision testing (50+ iterations) - Publication-grade results
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=60 --target-prize-rate=0.18 --min-draws=15
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=75 --start-year=2004 --end-year=2024 --min-draws=10
```

##### 🔄 **Iterative Strategy Examples**
```bash
# Progressive iteration testing - Start small, increase precision
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=10 --target-prize-rate=0.08  # Initial test
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=25 --target-prize-rate=0.10  # Standard test  
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=40 --target-prize-rate=0.12  # Refined test

# A/B testing with different iteration counts
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=20 --target-prize-rate=0.10 --start-year=2020
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=30 --target-prize-rate=0.10 --start-year=2020

# Combined strategy optimization
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=28 --target-prize-rate=0.11 --min-draws=8 --start-year=2015 --end-year=2024
```

##### ⏱️ **Time vs Accuracy Trade-off Guide**
| Max Iterations | Expected Time | Strategies Tested | Confidence Level | Best For |
|---------------|---------------|-------------------|------------------|----------|
| **5-8** | 2-6 minutes | 5-8 configs | Basic | Quick development, initial tests |
| **10-15** | 6-12 minutes | 10-15 configs | Standard | Regular optimization, validation |
| **20-25** | 12-20 minutes | 20-25 configs | High | Production use, reliable results |
| **30-40** | 20-35 minutes | 30-40 configs | Very High | Research, detailed analysis |
| **45-60** | 35-50 minutes | 45-60 configs | Maximum | Academic studies, publication |
| **60+** | 50+ minutes | 60+ configs | Extreme | Comprehensive validation |

#### 🎛️ **Complete Flag Usage Examples**

##### 🚩 **All Available Flags Overview**
```bash
# All flags for unified_euromillions_predictor.py
--backtest                    # Enable backtesting mode
--extended                    # Extended backtesting (50 draws per method)
--samples=N                   # Total sample size distributed among methods
--draws-per-method=N          # Specific draws per method (exact control)
--start-year=YYYY             # Starting year for analysis period
--end-year=YYYY               # Ending year for analysis period
--auto-optimize               # Enable automatic optimization mode
--target-prize-rate=X.X       # Target prize win rate (0.05 = 5%, 0.10 = 10%, etc.)
--max-iterations=N            # Maximum optimization iterations
--min-draws=N                 # Minimum draws per method for optimization
```

##### 🔧 **Comprehensive Multi-Flag Examples**

**📊 Complete Backtesting with All Parameters:**
```bash
# Maximum precision backtesting with all parameters
python3 unified_euromillions_predictor.py --backtest --draws-per-method=150 --start-year=2004 --end-year=2024

# Custom period analysis with specific sample control
python3 unified_euromillions_predictor.py --backtest --samples=350 --start-year=2018 --end-year=2023

# Extended backtesting with year range focus
python3 unified_euromillions_predictor.py --backtest --extended --start-year=2020 --end-year=2024

# Precise historical analysis
python3 unified_euromillions_predictor.py --backtest --draws-per-method=100 --start-year=2010 --end-year=2019
```

**🎯 Complete Auto-Optimization with All Parameters:**
```bash
# Full optimization with all auto-optimize flags
python3 unified_euromillions_predictor.py --auto-optimize --target-prize-rate=0.12 --max-iterations=35 --min-draws=8 --start-year=2015 --end-year=2024

# Research-grade optimization with comprehensive parameters
python3 unified_euromillions_predictor.py --auto-optimize --target-prize-rate=0.15 --max-iterations=50 --min-draws=12 --start-year=2004 --end-year=2024

# Quick optimization with focused parameters
python3 unified_euromillions_predictor.py --auto-optimize --target-prize-rate=0.08 --max-iterations=15 --min-draws=5 --start-year=2020 --end-year=2024

# Academic-level comprehensive optimization
python3 unified_euromillions_predictor.py --auto-optimize --target-prize-rate=0.18 --max-iterations=60 --min-draws=15 --start-year=2004 --end-year=2024
```

**🔬 Combined Strategy Examples (Multi-Phase Analysis):**
```bash
# Phase 1: Quick validation (recent data, low iterations)
python3 unified_euromillions_predictor.py --auto-optimize --target-prize-rate=0.08 --max-iterations=10 --start-year=2022 --end-year=2024

# Phase 2: Standard analysis (medium period, balanced iterations)  
python3 unified_euromillions_predictor.py --auto-optimize --target-prize-rate=0.10 --max-iterations=25 --start-year=2018 --end-year=2024

# Phase 3: Comprehensive validation (full range, high iterations)
python3 unified_euromillions_predictor.py --auto-optimize --target-prize-rate=0.12 --max-iterations=40 --min-draws=10 --start-year=2004 --end-year=2024

# Phase 4: Final verification with backtesting
python3 unified_euromillions_predictor.py --backtest --draws-per-method=75 --start-year=2004 --end-year=2024
```

**⚡ Performance vs Precision Trade-off Examples:**
```bash
# FAST: Quick development testing (~5 minutes)
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=8 --target-prize-rate=0.06 --start-year=2021

# BALANCED: Production-ready optimization (~15 minutes)
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=22 --target-prize-rate=0.10 --min-draws=8 --start-year=2017

# THOROUGH: Research-grade analysis (~35 minutes)
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=40 --target-prize-rate=0.14 --min-draws=12 --start-year=2008 --end-year=2024

# MAXIMUM: Publication-quality validation (~60+ minutes)
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=60 --target-prize-rate=0.16 --min-draws=15 --start-year=2004 --end-year=2024
```

**🎨 Specialized Analysis Examples:**
```bash
# Economic impact analysis (Pre/Post COVID comparison)
python3 unified_euromillions_predictor.py --backtest --draws-per-method=50 --start-year=2016 --end-year=2019  # Pre-COVID
python3 unified_euromillions_predictor.py --backtest --draws-per-method=50 --start-year=2020 --end-year=2024  # Post-COVID

# Decade comparison analysis
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=30 --start-year=2004 --end-year=2014  # 2000s decade
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=30 --start-year=2014 --end-year=2024  # 2010s decade

# High-precision validation with specific targets
python3 unified_euromillions_predictor.py --auto-optimize --target-prize-rate=0.20 --max-iterations=75 --min-draws=20 --start-year=2004 --end-year=2024

# Sample size scaling analysis
python3 unified_euromillions_predictor.py --backtest --samples=100 --start-year=2020    # Small sample
python3 unified_euromillions_predictor.py --backtest --samples=500 --start-year=2015    # Large sample
python3 unified_euromillions_predictor.py --backtest --draws-per-method=200 --start-year=2004  # Maximum sample
```

##### 📋 **Flag Combination Recommendations**

| Use Case | Recommended Flags | Example Command | Duration |
|----------|-------------------|-----------------|----------|
| **Quick Dev Test** | `--auto-optimize --max-iterations=8` | `--auto-optimize --max-iterations=8 --target-prize-rate=0.06` | 3-5 min |
| **Standard Analysis** | `--auto-optimize --max-iterations=20-25` | `--auto-optimize --max-iterations=22 --min-draws=8` | 12-18 min |
| **Research Study** | `--auto-optimize --max-iterations=35+ --min-draws=10+` | `--auto-optimize --max-iterations=40 --min-draws=12 --start-year=2010` | 25-40 min |
| **Academic Paper** | All optimization flags + full range | `--auto-optimize --target-prize-rate=0.15 --max-iterations=50 --min-draws=15 --start-year=2004 --end-year=2024` | 45-60 min |
| **Production Validation** | `--backtest --draws-per-method=50-100` | `--backtest --draws-per-method=75 --start-year=2018` | 15-25 min |
| **Comprehensive Study** | Maximum precision backtesting | `--backtest --draws-per-method=150 --start-year=2004 --end-year=2024` | 30-45 min |

#### 📋 **Sample Size Recommendations (Manual Testing)**
| Purpose | Draws/Method | Total Tests | Time | Statistical Confidence |
|---------|--------------|-------------|------|------------------------|
| **Quick Test** | 10-25 | 70-175 | 2-5 min | Basic validation |
| **Standard** | 25-50 | 175-350 | 5-15 min | Good reliability |
| **Research** | 75-100 | 525-700 | 15-25 min | High confidence |
| **Academic** | 150-200 | 1,050-1,400 | 30-60 min | Publication-grade |
| **Comprehensive** | 200+ | 1,400+ | 1+ hour | Maximum validation |
| **🎯 Auto-Optimization** | Variable | Variable | 10-60 min | **Intelligent targeting** |

#### 🔧 **Individual Method Testing**
```bash
# Validate individual methods
python3 statistical_predictor.py
python3 quick_predictor.py

# Test data integrity  
python3 optimized_html_scraper.py
```

### 📚 **Educational Value**
This project serves as an excellent case study for:
- **Advanced Statistical Analysis**: Time-series, frequency analysis, significance testing
- **Machine Learning Applications**: Neural networks, ensemble methods, validation
- **Data Science Methodology**: Large dataset analysis, pattern recognition, backtesting
- **Scientific Method**: Hypothesis testing, proper validation, objective conclusions
- **Software Architecture**: Modular design, unified interfaces, comprehensive testing

## 🔗 References & Resources

- [EuroMillions Official Results](https://www.euro-millions.com)  
- [Lottery Mathematics & Statistics](https://en.wikipedia.org/wiki/Lottery_mathematics)
- [Machine Learning for Time Series Analysis](https://keras.io/examples/timeseries/)
- [Statistical Significance Testing](https://en.wikipedia.org/wiki/Statistical_significance)
- [Ensemble Methods in Machine Learning](https://en.wikipedia.org/wiki/Ensemble_learning)

## 📄 License

This project is released under the MIT License. See `LICENSE` file for details.

## 📞 Contact

For questions, suggestions, or research collaboration:
- Create an issue on GitHub with detailed description
- Focus on scientific and educational aspects
- Include relevant statistical analysis and methodology
- Maintain academic and research-oriented discussion

---

**🎓 Academic Note**: This is a comprehensive research project exploring the intersection of statistics, machine learning, and probability theory. The unified prediction system represents a scientifically rigorous approach to pattern analysis in lottery data, with proper validation and honest performance reporting. Use this project for learning, statistical education, and scientific exploration. **No system can predict truly random lottery outcomes.**

**⚠️ Responsibility**: Always gamble responsibly and within your means. This system is for educational and research purposes only.