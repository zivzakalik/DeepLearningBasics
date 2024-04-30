# DeepLearningBasics
Home work assignment for deep learning class
This git repo contains 3 home work assignment for deep learning course.

## Simple fully connencted
#### The assignment
This project involves implementing neural network architectures to understand their functionality and effectiveness on image classification tasks. We focused on two types of architectures: a Fully Connected Network (FCN) and a Convolutional Neural Network (CNN), both trained on the MNIST dataset.

## The solution

### 1. Fully Connected Network (FCN)
The Fully Connected Network (FCN) implemented in this project serves as a fundamental example of neural networks, designed specifically for tasks like digit recognition using the MNIST dataset.

- **Input Layer**: Accepts flattened image pixels, translating each 28x28 image into a 784-element vector.
- **Hidden Layer**: Features a user-defined number of neurons (`hidden_size1`), all fully connected to the input. This layer allows the exploration of various activation functions such as sigmoid or ReLU, introducing non-linearity to the model.
- **Output Layer**: Comprises neurons equal to the number of target classes (10 for MNIST), each producing a score corresponding to a specific digit.
- **Loss Function**: Employs softmax cross-entropy for loss computation, crucial for the model's ability to handle multiple classes.
- **Backpropagation**: Demonstrates the fundamental mechanism of learning in neural networks through gradient descent, which updates weights and biases based on the computed gradients.

### 2. Convolutional Neural Network (CNN)
The `GroceryStoreCNN`, a more complex structure, is optimized for processing image data, capturing spatial hierarchies and dependencies effectively.

- **Layers**: Composed of several convolutional layers, each supplemented with batch normalization and ReLU activation to maintain non-linearity and training stability. Each convolutional layer is followed by max pooling to reduce dimensionality and highlight important features.
- **Fully Connected Layers**: After processing through the convolutional layers, the data is flattened and passed through fully connected layers. A dropout layer is included to mitigate overfitting by randomly ignoring some neurons during training, enhancing the model's generalization capabilities.
- **Output**: The network's final layer calculates logits for each class, subsequently transformed into a probability distribution via a log-softmax function, providing a clear, interpretable output for classification tasks.

These architectures illustrate two core approaches in neural network design—fully connected and convolutional—highlighting their unique capabilities and suitability for different types of data and tasks.

## NLP sentiment prediction
#### The assignment
The objective of this exercise is to develop a familiarity with recurrent neural networks (RNNs) by applying them to the task of emotion detection in text. Emotion detection is increasingly valuable for understanding sentiments in user-generated content, such as determining the tone of reviews or social media posts.

## README for Emotion Detection Assignment

### Overview
The objective of this exercise is to develop a familiarity with recurrent neural networks (RNNs) by applying them to the task of emotion detection in text. Emotion detection is increasingly valuable for understanding sentiments in user-generated content, such as determining the tone of reviews or social media posts.

The task involves detecting the emotion of a sentence with a goal of achieving at least 47% accuracy on the test set. Students are encouraged to experiment with different RNN architectures and techniques, including:
- Vanilla RNN
- Gated models such as GRU or LSTM
- Various optimization and regularization methods
- Different hyperparameter combinations

### Model Architectures
#### 1. SentimentRNN (Vanilla RNN)
This model utilizes a basic RNN structure with ReLU activation. The architecture is designed to understand the baseline performance of RNNs without gates.

- **Embedding Layer**: Maps words to a high-dimensional space, initialized with a pre-trained weights matrix.
- **RNN Layer**: Processes the embeddings through multiple RNN layers with dropout for regularization.
- **Output Layer**: A fully connected layer that outputs the probability distribution over the emotion classes using log softmax.

#### 2. SentimentLSTM (Gated RNN)
This model uses an LSTM (Long Short-Term Memory) network, which is effective in capturing long-term dependencies in text, crucial for understanding contextual nuances.

- **Embedding Layer**: Converts text into embeddings, initialized with a pre-trained weights matrix.
- **LSTM Layer**: Processes the embeddings through bidirectional LSTM layers with dropout to enhance the model's ability to learn from the data while preventing overfitting.
- **Output Layer**: Transforms the LSTM outputs into final emotion predictions using a fully connected layer and log softmax.


## VAE
#### The assignment
This project explores the concept of disentangled representations, particularly focusing on the challenge of representing both continuous and discrete variables in a dataset. Disentangled representations are crucial in understanding how changes in latent dimensions correlate with changes in individual data variabilities, while being invariant to other factors.

#### The solution
The solution is inspired by the paper "Learning Disentangled Joint Continuous and Discrete Representations" which employs a model known as JointVAE. This model is a type of Variational Autoencoder that has been extended to effectively capture both continuous and discrete generative factors in an unsupervised manner. This approach benefits from the inherent advantages of VAEs including stable training, diversity in generated samples, and a well-principled inference mechanism.

### Specifics of the Implementation
#### Discrete Features in Dataset
In our dataset, examples of discrete features include:
- Eyeglasses
- Bangs
- Face position (e.g., forward, left side, right side)
- Age categories
- Facial shape (e.g., oval)
- Skin tone
- Facial expressions (e.g., smile, neutral, frown)

#### Model Framework
The JointVAE framework integrates the modeling of both continuous and discrete latent variables:
- **Continuous Variables**: Represent aspects like position and scale.
- **Discrete Variables**: Encode categorical data such as presence of eyeglasses or type of facial expression.

#### Loss Function Details
The VAE loss function is modified to include:
- **Controlled Information Capacity**: Increases gradually during training, helping the model to stabilize the encoding of information into the latent space.
- **Regularization Term**: Influences the KL divergence to ensure that it matches a predefined capacity, improving the disentanglement and reconstruction quality.

#### Gumbel-Softmax Distribution
To handle discrete variables, the Gumbel-Softmax distribution is used, providing a differentiable approximation to the sampling from a categorical distribution. This is crucial for backpropagating errors through discrete variables.

### Model Architectures
#### Encoder
- Multiple convolutional layers with ReLU activations, reducing dimensionality while capturing relevant features.

#### Decoder
- Mirrors the encoder architecture but in reverse, reconstructing the input image from the latent space representation.

#### Bottleneck
- Handles both types of latent variables (continuous and discrete), with linear layers transforming them into an intermediate space.

### Experiments and Results
#### Models Developed
1. **Model 1**: Focused on evaluating the model without capacity constraints, revealing the importance of these constraints in managing latent space separations.
2. **Model 2**: Included capacity constraints with 10 discrete latent features, showing significant improvements in disentanglement and reconstruction accuracy.
3. **Model 3**: Extended the discrete latent space to 40 features to see if the model could handle higher complexity, which it did successfully.

#### Training Parameters
- **Batch Size**: 64
- **Discrete Dimensions**: [10, 40]
- **Continuous Dimension**: 32
- **Learning Rate**: 0.004
- **Optimizer**: Adam
- **Epochs**: 100
- **Capacity Ranges**: [30, 100]
- **Image Size**: 64x64

### Conclusion
This assignment demonstrated the effective use of VAEs in handling complex datasets with both continuous and discrete variables. Through careful architectural choices and parameter tuning, we successfully disentangled the latent representations, facilitating a deeper understanding of the underlying data structure.
