**ABOUT DATASET**
The dataset consists of face images from 31 distinct classes, each representing a well-known individual. It includes a diverse set of celebrity and public figure images, such as Zac Efron, Virat Kohli, Vijay Deverakonda, Tom Cruise, and Roger Federer, alongside icons from various domains like sports, acting, and music. For example, actors Robert Downey Jr., Priyanka Chopra, Natalie Portman, Hugh Jackman, and Jessica Alba are included, providing a range of face types, expressions, and poses. 

This dataset is valuable for training a face recognition model, as it incorporates diverse demographics, genders, and features, ensuring the model learns to recognize faces in varying conditions. With prominent figures like Charlize Theron, Dwayne Johnson, Brad Pitt, Alexandra Daddario, and others, the dataset allows for a robust learning experience that spans age groups and diverse facial characteristics. 

The mix of global celebrities and personalities in the dataset enhances its applicability to real-world scenarios, providing a foundation for models aiming for high accuracy in multi-class facial recognition tasks.

https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset/data

![datasets-for-facial-recognition](https://github.com/user-attachments/assets/26829938-8f44-48e5-a1d4-bd08ca008723)

**GOAL**


**ALGORİTHM**
we build a face recognition model using deep learning. Specifically, we use a Convolutional Neural Network (CNN) to process images, as CNNs are well-suited for image recognition tasks due to their ability to automatically learn spatial hierarchies in images.

Here's a breakdown of the key algorithms and components used:

Convolutional Neural Networks (CNNs):

CNNs are specialized neural networks primarily designed for image data. They use multiple layers of "convolution" filters to detect patterns in images, such as edges, textures, and eventually complex shapes. The CNN in this model starts with VGG16, a popular deep CNN architecture that has been pre-trained on the ImageNet dataset. The pre-trained VGG16 model captures a wide range of useful features, which we adapt to our specific task of face recognition.
Transfer Learning:

Transfer learning leverages a pre-trained model (VGG16 here) as a starting point. Since the VGG16 model has already learned general image features, we don't have to start training from scratch. We "freeze" the initial layers of VGG16 to retain these pre-trained features and add a custom output layer specifically designed to recognize faces in our dataset. This approach speeds up training and often leads to higher accuracy, especially with limited data.
Data Augmentation:

The ImageDataGenerator class in Keras helps improve model generalization by applying random transformations to training images, such as rescaling, rotation, and flipping. Data augmentation is crucial when we have limited data, as it allows the model to learn variations in the dataset and become more robust to different orientations and lighting.
Early Stopping:

The EarlyStopping callback is used to monitor the model’s validation loss and stop training if it does not improve over a certain number of epochs. This helps prevent overfitting, where the model might start memorizing the training data instead of generalizing well to new data. The restore_best_weights=True parameter ensures that the model returns to the state with the lowest validation loss.
Fine-Tuning with Fully Connected Layers:

The top of the model includes fully connected (dense) layers. After flattening the output from VGG16, we add a dense layer with 256 neurons and a Dropout layer to reduce overfitting. The final output layer has a softmax activation function, allowing the model to output probabilities for each class (face), which is suitable for our multi-class classification problem.




**RESSULT**

Accuracy: 0.9698: This value indicates that the model achieved an impressive accuracy of approximately 96.98% on the training dataset. This means that out of all the training samples, about 97 out of 100 were correctly classified. Such high accuracy suggests that the model effectively learned to recognize the faces in the training data.

Loss: 0.2895: The loss value quantifies how well the model's predictions match the actual labels during training. A loss of 0.2895 indicates that the model's predictions are relatively close to the true values, as lower loss values are better. This suggests that the model is making accurate predictions and has learned meaningful features from the training data.

Validation Accuracy: 0.8088: This value represents the model's performance on the validation dataset, achieving an accuracy of approximately 80.88%. This means that about 81 out of 100 validation samples were correctly classified. While this is a good accuracy rate, it is significantly lower than the training accuracy, indicating that the model may not generalize as well to unseen data.

Validation Loss: 0.7292: The validation loss of 0.7292 is higher than the training loss, suggesting that the model's performance on the validation dataset is not as strong as on the training dataset. This discrepancy might indicate that the model is overfitting, meaning it has learned the training data too well, including noise and specific details, which makes it less effective on new, unseen data.

Conclusion
Overall, while the model demonstrates excellent performance on the training set, with a high accuracy and low loss, the validation results suggest that further improvements are needed to enhance its generalization capability. It may be beneficial to implement techniques such as regularization, dropout layers, or data augmentation to reduce overfitting and improve performance on the validation set. This balance is crucial for developing a robust face recognition model that performs well in real-world applications.

![billiellish true](https://github.com/user-attachments/assets/4f76955c-7222-442c-b5e9-560c87a38629)


