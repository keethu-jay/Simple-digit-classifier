# import libraries
from flask import Flask, render_template, request, jsonify 
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
    # convert to numpy array
    image = np.array(image)
    # normalize pixel values to [0, 1]
    image = image / 255
    # flatten the image
    image = image.reshape(1, 28*28)
    return image

# Debug prints
    print("Processed image shape:", image.shape)
    print("Processed image sample:", image[0, :10])  # Print first 10 values as a sample
    

# create home page
@app.route('/')
def home():
    return render_template('index.html')

# create page to upload image
@app.route('/predict', methods=['POST'])
def predict():
    # checks to see if user uploaded a file
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    # retrieves the uplloaded file from the request
    file = request.files['file']
    # check if the file has a valid filename
    if file.filename == '':
        return jsonify({'error': 'no filename'}), 400
    
    # opens up the image for processing
    image = Image.open(file.stream)
    # prepares the image using the function above
    prepared_image = prepare_image(image)

    # passes prepared image to the model
    prediction = model.predict(prepared_image)
    # get the prediction
    predicted_label = np.argmax(prediction)

    # return label as JSON response
    return jsonify({'prediction': int(predicted_label)})

# run application
if __name__ == '__main__':
    app.run(debug=True)


