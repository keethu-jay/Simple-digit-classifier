# import libraries
from flask import Flask, render_template, request, jsonify 
import tensorflow as tf
from tensorflow import keras 
import numpy as np
from PIL import Image, ImageOps


# initialize Flask application
app = Flask(__name__)

# load trained model
model = keras.models.load_model('/Users/keerthanajayamoorthy/Desktop/Digit Classifier Project/model.h5')

# formatting for input image
def prepare_image(image):
    # convert to grayscale
    image = ImageOps.grayscale(image)
    # Resize to 28x28
    image = image.resize((28,28))
    # convert to numpy array and normalize
    image = np.array(image).astype('float32') / 255.0
    # reshape to match the models input shape
    image = image.reshape(1, 28, 28, 1)
    return image
    

# create home page
@app.route('/')
def home():
    return render_template('index.html')

# create page to upload image
@app.route('/predict', methods=['POST'])
def predict():
    # checks to see if user uploaded a file
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400
    # retrieves the uplloaded file from the request
    file = request.files['file']
    # check if the file has a valid filename
    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400
    
    try:
        # opens up the image for processing
        image = Image.open(file.stream)
        # prepares the image using the function above
        prepared_image = prepare_image(image)
        # passes prepared image to the model
        predictions = model.predict(prepared_image)
        # get the prediction
        predicted_label = np.argmax(predictions[0])
        probabilities = predictions[0]

        # Log results for debugging
        print(f"Predicted class: {predicted_label}")
        print(f"Probabilities: {probabilities}")
        # Prepare response
        response = {
            'prediction': int(predicted_label),
            'probabilities': [float(f"{prob:.4f}") for prob in probabilities]
        }
        
        # return label as JSON response
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# run application
if __name__ == '__main__':
    app.run(debug=True)


