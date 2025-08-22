# EuroGuesser 🎯

A comprehensive EuroMillions lottery prediction system that explores multiple approaches to number generation, from statistical analysis to advanced AI techniques and physical bias detection.

## 🎲 Overview

EuroGuesser is an experimental project that implements various prediction strategies for EuroMillions lottery numbers. While acknowledging that lottery draws are designed to be random, this system explores legitimate approaches including statistical analysis, machine learning, and physical bias detection that could theoretically provide insights into lottery patterns.

## 🌟 Features

### 📊 Statistical Prediction Models
- **Quick Predictor**: Fast statistical analysis using frequency patterns
- **Advanced Statistical Predictor**: Comprehensive analysis with gap detection and pattern recognition
- **Prediction Summary**: Overview of all available models and dataset statistics

### 🤖 Machine Learning & AI
- **TensorFlow Neural Networks**: LSTM, CNN, and Dense models for pattern recognition
- **Ensemble Methods**: Multiple model architectures working together
- **Advanced AI Techniques**: Quantum-inspired algorithms, chaos theory, and Fibonacci patterns

### 🔬 Physical Bias Detection
- **Equipment Bias Analysis**: Detect potential ball weight irregularities
- **Positional Bias Detection**: Analyze if certain numbers prefer specific draw positions
- **Machine Change Detection**: Identify periods of equipment maintenance or replacement
- **Environmental Analysis**: Seasonal patterns and temperature/humidity effects

### 📈 Data Management
- **Historical Dataset**: 1,869 EuroMillions draws from 2004-2025
- **Data Crawler**: Automatic updates from euro-millions.com
- **Multiple Formats**: CSV and training-ready data formats

## 🚀 Quick Start

### Prerequisites
```bash
# Core dependencies (available by default)
python3 >= 3.8
numpy >= 1.26.4
pandas >= 2.1.4
matplotlib >= 3.6.3

# Optional ML dependencies
pip install tensorflow scikit-learn
```

### Basic Usage

#### 1. Quick Statistical Predictions
```bash
python3 quick_predictor.py
```

#### 2. Comprehensive Analysis
```bash
python3 statistical_predictor.py
```

#### 3. Physical Bias Detection
```bash
python3 physical_bias_detector.py
```

#### 4. Advanced AI Methods
```bash
python3 advanced_ai_predictor.py
```

#### 5. TensorFlow Neural Networks
```bash
python3 tensorflow_euromillions_predictor.py
```

## 📁 Project Structure

```
euroguesser/
├── README.md                              # This file
├── CLAUDE.md                              # Development guidelines
├── euromillions_historical_results.csv   # Complete dataset (1,869 draws)
├── euromillions_training_data.txt         # Training format data
│
├── Statistical Models/
│   ├── quick_predictor.py                 # Fast statistical predictions
│   ├── statistical_predictor.py           # Advanced statistical analysis
│   └── prediction_summary.py              # Models overview
│
├── Machine Learning/
│   ├── tensorflow_euromillions_predictor.py  # TensorFlow implementation
│   ├── improved_euromillions_predictor.py    # Ensemble methods
│   ├── advanced_ai_predictor.py              # Cutting-edge AI techniques
│   ├── model_training.py                     # Original neural network
│   └── predictions.py                        # Model predictions
│
├── Physical Analysis/
│   └── physical_bias_detector.py          # Equipment bias detection
│
├── Data Management/
│   ├── euromillions_crawler.py           # Data collection
│   └── improved_training.py              # Enhanced training methods
│
└── Legacy/
    ├── euromilhoes-python.py             # Sequential matching
    └── LotteryAi.py                       # Alternative training
```

## 🎯 Prediction Strategies

### 1. Statistical Approaches
- **Frequency Analysis**: Most/least common numbers
- **Gap Analysis**: Numbers "due" for appearance
- **Pattern Recognition**: Consecutive numbers, odd/even distribution
- **Hot/Cold Numbers**: Recent vs. historical performance

### 2. Machine Learning
- **LSTM Networks**: Sequential pattern learning
- **CNN Models**: Pattern recognition in number sequences
- **Dense Networks**: Traditional neural network approaches
- **Ensemble Methods**: Combining multiple models with different architectures

### 3. Advanced AI Techniques
- **Quantum-Inspired Sampling**: Probabilistic superposition principles
- **Chaos Theory**: Lorenz attractor-based generation
- **Fibonacci Patterns**: Golden ratio and natural sequences
- **Temperature Sampling**: Controlled randomness for diversity

### 4. Physical Bias Detection
- **Ball Weight Analysis**: Detect heavier/lighter balls
- **Positional Bias**: Numbers preferring certain draw positions
- **Equipment Changes**: Machine maintenance detection
- **Environmental Factors**: Seasonal and weather-related patterns

## 📊 Dataset Information

- **Total Draws**: 1,869 historical results
- **Date Range**: February 2004 - August 2025
- **Format**: Date, Main1-5, Star1-2
- **Source**: Official EuroMillions results
- **Update Method**: Automated web crawler

## 🔬 Scientific Approach

### Legitimate Research Areas
1. **Physical Equipment Biases**: Real lottery machines can develop mechanical biases
2. **Statistical Anomalies**: Detecting non-random patterns in historical data
3. **Environmental Factors**: Temperature/humidity effects on equipment
4. **Maintenance Cycles**: Equipment replacement and ball set changes

### Mathematical Limitations
- Each draw is independent (past results don't influence future draws)
- True randomness cannot be predicted mathematically
- Probability remains 1 in 139,838,160 for any specific combination
- No algorithm can overcome fundamental randomness

## 📈 Results & Findings

### Statistical Insights
- Most frequent main number: **23** (214 appearances)
- Least frequent main number: **22** (148 appearances)
- Most frequent star: **3** (373 appearances)
- Average main numbers: **25.5**

### Physical Bias Detection
- **10 main numbers** show >10% frequency deviation
- **5 star numbers** show >15% frequency deviation
- **8 periods** show evidence of equipment changes
- **Positional biases** detected (numbers 1-5 favor smallest position)

### Model Performance
- Neural networks achieve convergence but struggle with diversity
- Ensemble methods provide better prediction variety
- Physical bias detection offers most scientifically sound approach
- Statistical methods provide good baseline performance

## 🚨 Important Disclaimers

### ⚠️ Legal & Ethical
- **For research and entertainment purposes only**
- **No guarantee of improved winning odds**
- **Lottery outcomes are designed to be unpredictable**
- **Gambling responsibly - never bet more than you can afford to lose**

### 🔬 Scientific Limitations
- Detected patterns may be statistical noise
- Modern lottery equipment minimizes physical biases
- Regular maintenance reduces equipment irregularities
- True randomness cannot be overcome by any method

## 🛠️ Development

### Contributing
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Guidelines
- Follow scientific methodology
- Document statistical significance
- Include proper disclaimers
- Focus on educational value
- Maintain code quality

### Testing
```bash
# Run statistical tests
python3 -m pytest tests/

# Validate data integrity
python3 validate_dataset.py

# Check model performance
python3 evaluate_models.py
```

## 📚 Educational Value

This project serves as an excellent case study for:
- **Statistical Analysis**: Frequency analysis, chi-square tests, significance testing
- **Machine Learning**: Neural networks, ensemble methods, time series analysis
- **Data Science**: Large dataset analysis, pattern recognition, visualization
- **Scientific Method**: Hypothesis testing, validation, proper conclusions
- **Probability Theory**: Understanding randomness, bias detection, mathematical limits

## 🔗 References & Resources

- [EuroMillions Official Results](https://www.euro-millions.com)
- [Statistical Analysis of Lottery Games](https://en.wikipedia.org/wiki/Lottery_mathematics)
- [Physical Bias in Random Number Generation](https://www.random.org/essay.html)
- [Machine Learning for Time Series](https://keras.io/examples/timeseries/)

## 📄 License

This project is released under the MIT License. See `LICENSE` file for details.

## 📞 Contact

For questions, suggestions, or research collaboration:
- Create an issue on GitHub
- Focus on scientific and educational aspects
- Include relevant statistical analysis

---

**Remember**: This is an experimental research project. Lottery games are designed to be random, and no system can guarantee winning numbers. Use this project for learning, entertainment, and scientific exploration only. Always gamble responsibly.