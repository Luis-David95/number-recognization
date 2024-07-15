import base64
import io
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image

mnist = tf.keras.datasets.mnist

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

def print_image(raw_image_data):
    """Print image with '+' & '-'"""
    one_sample_image = np.array(raw_image_data).reshape([28, 28])
    for index_a in range(len(one_sample_image)):
        for index_b in range(len(one_sample_image[index_a])):
            if index_b == 27:
                print('')
            point = one_sample_image[index_a][index_b]
            if point > 0.3:
                print("+", end="")
            elif point > 0 and point <= 0.3:
                print("-", end="")
            else:
                print(" ", end="")

def test_image(raw_image):
    np_img = np.array(raw_image).reshape(1, 784)  # Reshape for the model
    predictions = model.predict(np_img)
    return np.argmax(predictions)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', title='Recognize Number')

@app.route('/recog', methods=['POST'])
def upload_img():
    img_str = request.form['img'][22:]
    img_data = base64.b64decode(img_str)
    img_obj = Image.open(io.BytesIO(img_data))
    img_obj = img_obj.resize((28, 28)).convert('L')  # Convert to grayscale
    img_arr = np.array(img_obj).flatten() / 255.0  # Flatten and normalize

    print_image(img_arr)
    num = test_image(img_arr)

    return jsonify(ok=1, num=int(num))

if __name__ == '__main__':
    app.run()