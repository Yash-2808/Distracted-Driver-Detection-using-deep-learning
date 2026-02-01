import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

print("Converting Keras model to TensorFlow Lite...")

# Load the original Keras model
model_path = 'v7_plus_distracted_driver.keras'
print(f"Loading model from {model_path}...")

try:
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Convert to TensorFlow Lite with optimizations
print("\nConverting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimizations for smaller size and faster inference
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # Use float16 for smaller size

# Convert the model
try:
    tflite_model = converter.convert()
    print("Conversion successful!")
except Exception as e:
    print(f"Error during conversion: {e}")
    # Fallback without optimizations
    print("Trying conversion without optimizations...")
    converter.optimizations = []
    converter.target_spec.supported_types = []
    tflite_model = converter.convert()
    print("Fallback conversion successful!")

# Save the TFLite model
tflite_path = 'v7_plus_distracted_driver.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved to {tflite_path}")

# Compare model sizes
import os
keras_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB

print(f"\nModel Size Comparison:")
print(f"Original Keras model: {keras_size:.2f} MB")
print(f"TFLite model: {tflite_size:.2f} MB")
print(f"Size reduction: {((keras_size - tflite_size) / keras_size * 100):.1f}%")

# Test the TFLite model
print("\nTesting TFLite model...")
try:
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"TFLite Input shape: {input_details[0]['shape']}")
    print(f"TFLite Output shape: {output_details[0]['shape']}")
    
    # Create a test input
    input_shape = input_details[0]['shape']
    test_input = np.random.random_sample(input_shape).astype(np.float32)
    
    # Test prediction
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"TFLite prediction successful! Output shape: {output.shape}")
    print(f"Sample output: {output[0][:5]}")  # First 5 values
    
except Exception as e:
    print(f"Error testing TFLite model: {e}")

print("\nConversion complete! Your TFLite model is ready for deployment.")
