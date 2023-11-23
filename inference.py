import tensorflow as tf
import numpy as np
from data_prep import load_and_preprocess_image


def model_inference(image_path, model, input_shape, threshold):
    # Load and preprocess the image
    processed_image = load_and_preprocess_image(image_path, input_shape)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

    # Perform inference
    predictions = model.predict(processed_image)

    return (predictions > threshold).astype(np.float32)


if __name__ == "__main__":
    # Replace 'model_path' with the path to your saved model file
    model_path = ''

    input_shape = (768, 768) #Model input shape
    # Load the trained model
    loaded_model = tf.keras.models.load_model(model_path)

    # Example inference using an image path
    image_path_to_predict = ''

    best_treshold = 0.5 # we could find best th after training by evaluating perfomance on different th

    predictions = model_inference(image_path_to_predict, loaded_model, input_shape, best_treshold)

    # Use predictions as needed
    print("Predictions:", predictions)
