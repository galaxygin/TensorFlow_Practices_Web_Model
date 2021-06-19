from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflowjs as tfjs

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=3)
# test_loss, test_acc = model.evaluate(test_images, test_labels)

prediction = model.predict(test_images)

tfjs.converters.save_keras_model(model=model, artifacts_dir="export_fashion_classify")

for i in range(2):
    plt.grid(True)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel('Actual: ' + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
