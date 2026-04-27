import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os
import numpy as np

def main():
    print("=== Virtual Technologies Internship: AI Task 1 - Image Classification Model (CNN) ===")
    
    # 1. Load Fashion MNIST dataset
    print("Loading Fashion MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

    # Normalize pixel values
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    # Reshape for CNN
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print(f"Training data shape: {train_images.shape}")

    # 2. Build the CNN model
    print("Building the CNN architecture...")
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # 3. Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. Train the model
    print("\nTraining the model (this will take a moment)...")
    history = model.fit(train_images, train_labels, epochs=3, 
                        validation_data=(test_images, test_labels))

    # 5. Evaluate
    print("\nEvaluating on test data...")
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(f"Test Accuracy: {test_acc:.4f}")

    # 6. Visualizations
    print("\nGenerating visualizations...")
    os.makedirs('visualizations', exist_ok=True)
    
    # Accuracy Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    viz_path = os.path.join('visualizations', 'fashion_cnn_history.png')
    plt.savefig(viz_path)
    print(f"Saved training history to {viz_path}")
    
    # Sample predictions plot
    predictions = model.predict(test_images[:5])
    plt.figure(figsize=(12, 4))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
        predicted_label = class_names[np.argmax(predictions[i])]
        true_label = class_names[test_labels[i]]
        color = 'blue' if predicted_label == true_label else 'red'
        plt.xlabel(f"Pred: {predicted_label}\nTrue: {true_label}", color=color)
        
    pred_path = os.path.join('visualizations', 'fashion_cnn_predictions.png')
    plt.savefig(pred_path)
    print(f"Saved sample predictions to {pred_path}")
    
    print("=== AI Task 1 Completed Successfully ===\n")

if __name__ == "__main__":
    main()
