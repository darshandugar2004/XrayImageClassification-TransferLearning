Medical Image Classification with CNN
This project demonstrates a Convolutional Neural Network (CNN) for the classification of medical images, specifically a brain dataset. The model is built using TensorFlow and Keras, employing best practices such as data augmentation, batch normalization, and a learning rate scheduler to achieve high performance and robustness.

🚀 The Approach
The solution is centered around a custom-built CNN model designed to effectively learn from image data. The key components of the approach are:

Data Augmentation: To prevent overfitting and create a more robust model, we use extensive data augmentation. This process creates new training images by applying various transformations to the original dataset, including rotations, shifts, zooms, shearing, and flips. This ensures the model generalizes well to unseen data.

Model Architecture: The CNN architecture consists of multiple convolutional blocks, each followed by a Batch Normalization layer and a Max Pooling layer. This design allows the model to progressively extract more complex features from the input images while maintaining computational efficiency. The network concludes with fully connected layers and a Dropout layer to further combat overfitting.

Optimization & Training: The model is trained using the Adam optimizer with a categorical_crossentropy loss function. To fine-tune the training process, we utilize two key callbacks: ReduceLROnPlateau and ModelCheckpoint. The ReduceLROnPlateau callback automatically decreases the learning rate when the validation loss plateaus, helping the model converge to a better solution. The ModelCheckpoint saves the model's weights only when there is an improvement in the validation loss, ensuring that the final saved model is the best performing version.

📊 Evaluation & Results
The model's performance is evaluated on a dedicated validation set, providing a reliable measure of its classification ability.

Key Metrics
Validation Accuracy: 85.95%

Validation Loss: The best model achieved a validation loss of 0.31896.

Detailed Classification Report
A detailed classification report provides per-class metrics, including precision, recall, and F1-score.
