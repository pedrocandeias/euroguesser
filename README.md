# EuroGuesser - EuroMillions Prediction System

🎯 **Comprehensive EuroMillions lottery prediction system** with statistical analysis, machine learning, and N-key generation functionality.

## 🚀 Quick Start

```bash
# Main prediction system (recommended)
python3 unified_euromillions_predictor.py

# Generate 20 prediction keys and save to file
python3 unified_euromillions_predictor.py --num-keys=20 --save-to-file

# Test predictions against a specific key
python3 unified_euromillions_predictor.py --num-keys=10 --test-key=7,14,23,35,42:3,9

# Get help with all options
python3 unified_euromillions_predictor.py --help
```

## 📁 Project Structure

```
euroguesser/
├── 🚀 unified_euromillions_predictor.py    # Main system (start here!)
├── 📁 core/                                # Core prediction utilities  
├── 📁 models/                              # Machine learning components
├── 📁 data/                                # Dataset files
├── 📁 docs/                                # 📚 Full documentation (README.md here)
├── 📁 predictions/                         # Generated prediction files
└── 📁 euromillions_env/                    # Python virtual environment
```

## 📚 Full Documentation

👉 **For complete documentation, examples, and advanced usage, see:**  
**[docs/README.md](docs/README.md)** - Complete user guide with all features

## 🎯 Key Features

- **N Prediction Keys**: Generate configurable number of prediction keys
- **File Operations**: Save predictions to JSON, load and test against keys
- **Multiple Methods**: Statistical, ML, temporal, and bias detection
- **Backtesting**: Historical validation with official EuroMillions scoring
- **Auto-Optimization**: Intelligent parameter tuning

## 💡 Quick Examples

```bash
# Generate 5 keys with comprehensive analysis
python3 unified_euromillions_predictor.py --num-keys=5

# Save and test workflow
python3 unified_euromillions_predictor.py --num-keys=15 --save-to-file
python3 unified_euromillions_predictor.py --load-and-test=predictions/filename.json:1,12,23,34,45:2,8

# Backtesting and optimization
python3 unified_euromillions_predictor.py --backtest --extended
python3 unified_euromillions_predictor.py --auto-optimize --max-iterations=30
```

## 🛠️ Individual Components

If you want to use individual components:

```bash
# Statistical analysis only
python3 core/statistical_predictor.py

# Quick predictions
python3 core/quick_predictor.py  

# Data scraping
python3 core/optimized_html_scraper.py

# Model training
python3 models/model_training.py
```

---

⚠️ **For educational and research purposes only.** No system can predict truly random lottery outcomes.