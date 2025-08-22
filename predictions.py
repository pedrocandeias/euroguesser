import numpy as np
import tensorflow as tf

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def load_input_data(filepath):
    # Load historical lottery values from a file
    data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
    return data

def prepare_input(data):
    data[:, :5] = data[:, :5] / 50.0  # Normalize main numbers
    data[:, 5:] = data[:, 5:] / 12.0  # Normalize star numbers
    return data

def make_predictions(model, input_data):
    predictions = model.predict(input_data)
    return predictions

def interpret_predictions(predictions):
    # Take the latest prediction (last row) for generating new numbers
    latest_pred = predictions[-1]
    
    # Denormalize predictions back to original ranges
    main_pred = latest_pred[:5] * 50.0
    star_pred = latest_pred[5:] * 12.0
    
    # Convert to integers and ensure they're in valid ranges
    main_numbers = np.clip(np.round(main_pred).astype(int), 1, 50)
    star_numbers = np.clip(np.round(star_pred).astype(int), 1, 12)
    
    # Ensure no duplicates in main numbers
    main_numbers = list(main_numbers)
    while len(set(main_numbers)) < 5:
        for i, num in enumerate(main_numbers):
            if main_numbers.count(num) > 1:
                main_numbers[i] = np.random.randint(1, 51)
    
    # Ensure no duplicates in star numbers
    star_numbers = list(star_numbers)
    while len(set(star_numbers)) < 2:
        for i, num in enumerate(star_numbers):
            if star_numbers.count(num) > 1:
                star_numbers[i] = np.random.randint(1, 13)
    
    return sorted(main_numbers), sorted(star_numbers)

def main():
    model_path = 'lottery_model.keras'
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    historical_data_path = 'scraped_euromillions_training.txt'
    print(f"Loading historical data from {historical_data_path}...")
    input_data = load_input_data(historical_data_path)
    input_data = prepare_input(input_data)
    print(f"Loaded {len(input_data)} historical draws")
    
    print("Generating prediction...")
    predictions = make_predictions(model, input_data)
    main_numbers, star_numbers = interpret_predictions(predictions)
    
    print("\n" + "="*50)
    print("NEURAL NETWORK PREDICTION")
    print("="*50)
    print(f"Main Numbers: {main_numbers}")
    print(f"Star Numbers: {star_numbers}")
    print("="*50)


if __name__ == "__main__":
    main()