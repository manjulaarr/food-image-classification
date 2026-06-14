import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Food categories
CATEGORIES = ['burger', 'pizza', 'sushi', 'pasta', 'salad',
              'ice_cream', 'steak', 'tacos', 'ramen', 'biryani']

def build_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model():
    model = build_model(len(CATEGORIES))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Model Summary:")
    model.summary()
    print(f"\nModel built successfully for {len(CATEGORIES)} food categories")
    return model

def predict(model, image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    predicted_class = CATEGORIES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    print(f"Predicted: {predicted_class} ({confidence:.2f}% confidence)")
    return predicted_class

if __name__ == "__main__":
    model = train_model()