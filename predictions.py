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
    # Assuming the first 5 predictions are for the main numbers and the last 2 for the stars
    main_numbers_predictions = predictions[:, :5]
    stars_predictions = predictions[:, 5:]
    
    # Convert predictions to actual numbers (example heuristic)
    main_numbers = np.argsort(-main_numbers_predictions, axis=1)[:, :5] + 1
    star_numbers = np.argsort(-stars_predictions, axis=1)[:, :2] + 1
    
    return main_numbers, star_numbers

def main():
    model_path = 'lottery_model.keras'
    model = load_model(model_path)
    
    historical_data_path = 'euro_millions_entries_training.txt'
    input_data = load_input_data(historical_data_path)
    input_data = prepare_input(input_data)
    
    predictions = make_predictions(model, input_data)
    main_numbers, star_numbers = interpret_predictions(predictions)
    for i, (mains, stars) in enumerate(zip(main_numbers, star_numbers), 1):
        print(f"Prediction {i}: Main Numbers - {mains}, Star Numbers - {stars}")


if __name__ == "__main__":
    main()