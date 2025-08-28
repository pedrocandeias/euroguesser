# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive EuroMillions lottery prediction system with multiple prediction approaches including statistical analysis, machine learning, and deep neural networks. The project uses 20+ years of historical data (1,869 draws from 2004-2025) for accurate predictions.

## Core Architecture

The project consists of several prediction systems:

### 1. **Comprehensive Dataset**:
   - `euromillions_historical_results.csv`: Complete historical dataset with 1,869 draws (2004-2025)
   - `euromillions_training_data.txt`: Training format (compatible with existing models)
   - `euromillions_crawler.py`: Web crawler to fetch latest results from euro-millions.com

### 2. **Statistical Prediction Models** (Recommended - No Dependencies):
   - `statistical_predictor.py`: Advanced statistical analysis using frequency, patterns, gap analysis
   - `quick_predictor.py`: Fast predictions using multiple statistical approaches
   - `prediction_summary.py`: Overview and usage guide for all models

### 3. **Machine Learning Components** (Original):
   - `model_training.py`: Creates and trains Sequential neural network model
   - `LotteryAi.py`: Alternative training implementation
   - `predictions.py`: Loads trained model and generates predictions
   - `lottery_model.keras`: Trained model file

### 4. **Advanced ML Models** (Requires TensorFlow):
   - `advanced_euromillions_predictor.py`: Deep learning with LSTM, ensemble methods
   - `improved_training.py`: Enhanced version of original model with better preprocessing

### 5. **Legacy Components**:
   - `euromilhoes-python.py`: Sequential matching algorithm (brute-force approach)
   - `euro_millions_entries.txt`: Original smaller dataset
   - `euro_millions_entries_training.txt`: Training subset

## Development Commands

### Main System (Recommended)
```bash
python3 unified_euromillions_predictor.py
```
Complete unified prediction system with all methods and N-key functionality

### Individual Prediction Commands (No Dependencies Required)
```bash
python3 core/statistical_predictor.py
```
Comprehensive statistical analysis and prediction using frequency analysis, patterns, and gap analysis

```bash
python3 core/quick_predictor.py
```
Fast predictions using multiple statistical approaches with comparison

```bash
python3 core/prediction_summary.py
```
Show overview of all available models and dataset statistics

### Data Collection
```bash
python3 core/optimized_html_scraper.py
```
Update the historical dataset by crawling latest results from euro-millions.com

### Machine Learning Training (Requires TensorFlow)
```bash
python3 models/improved_training.py
```
Enhanced model training with better preprocessing and evaluation

```bash
python3 models/model_training.py
```
Original neural network training (saves model as `models/lottery_model.keras`)

```bash
python3 models/predictions.py
```
Generate predictions using trained models

## Data Format and Normalization

- Main numbers: 1-50 (normalized by dividing by 50.0)
- Star numbers: 1-12 (normalized by dividing by 12.0)
- Input format: CSV with 7 columns per line
- The model uses autoencoder architecture (input reconstruction as training target)

## Model Architecture

- Input layer: 7 features (5 main numbers + 2 stars)
- Hidden layers: 2 x Dense(64, relu activation)
- Output layer: Dense(7, sigmoid activation)
- Optimizer: Adam
- Loss function: Mean Squared Error

## Environment Requirements

### Core Dependencies (Available)
- Python 3.12.3
- NumPy 1.26.4
- Pandas 2.1.4

### Statistical Models (statistical_predictor.py, quick_predictor.py)
- No additional dependencies required
- Works with core dependencies only

### ML Models (advanced_euromillions_predictor.py, improved_training.py)
- TensorFlow/Keras (requires installation)
- Scikit-learn (requires installation)
- Matplotlib (for visualization)

### Dataset Information
- **Primary Dataset**: `data/scraped_euromillions_results.csv` (1,642 draws, 2004-2025)
- **Training Dataset**: `data/scraped_euromillions_training.txt` (ML training format)
- **Data Format**: CSV with Date,Main1,Main2,Main3,Main4,Main5,Star1,Star2
- **Training Format**: Simple CSV without headers (main1,main2,main3,main4,main5,star1,star2)