# Identification-of-medicinal-plants-using-CNN-Model
Identification of Different Medicinal Plants/Raw materials through Image Processing Using  Machine Learning Algorithms

 1. Collected the dataset of images containing different 
medicinal leaves.
 2. Preprocessed the collected images to enhance their 
quality and remove any noise.
 3. Splitting the dataset into training and testing sets. This 
will be used to train the ML model.
 4. CNN model is used for the image recognition 
and classification.
 5. The performance of the trained model is evaluated based 
on the testing data.
 6. Common evaluation metrics for image classification 
include accuracy.
 7. Once the model is trained and evaluated, we can use it to 
predict the class labels of new unseen images of the 
plants

TECHNOLOGY STACK USED:
Machine Learning
 ⮚ Convolutional Neural Network
 ⮚ Google Collab/Jupyter Notebook
 ⮚ HTML 5 
 ⮚ Python 3.9
 ⮚ Flask

Layers description of the model
1. Convolutional Layer
Filters/Kernels: Small-sized matrices (e.g., 3x3, 5x5) that slide over the input data to detect features. Each filter extracts a particular feature, like edges or textures.
Stride: number of pixels by which the filter moves over the input image. A stride of 1 moves the filter one pixel at a time, while a stride of 2 moves it two pixels at a time.
Padding: Adding zeros around the border of the input image to maintain the spatial dimensions after convolution.

2. Activation Function (ReLU) Used in this model
ReLU (Rectified Linear Unit): 
𝑓(𝑥)=max(0,𝑥)
f(x)=max(0,x). It introduces non-linearity into the model, enabling it to learn complex patterns.

3. Pooling Layer
Max Pooling: Taken the maximum value from each patch of the feature map (usually 2x2) to reduce spatial dimensions and computational load.
Average Pooling: Taken the average value from each patch of the feature map.

4. Fully Connected Layer
Flattening: Converting the 2D feature maps into a 1D vector to feed into fully connected layers.
Dense Layers: Traditional neural network layers where each neuron is connected to every neuron in the previous layer. These layers combine the features extracted by convolutional layers to make final predictions.

5. Output Layer
Softmax Activation: Used in the output layer for classification tasks, it converts the raw output scores into probabilities, indicating the likelihood of each class.


