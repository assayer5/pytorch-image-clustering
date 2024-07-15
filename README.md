## Project: Image Clustering

### Overview
Clustered images of numbers 5, 6, 8, and 9 from MNIST dataset. Tested two methods: feature extraction from pre-trained model, feature generation with autoencoders.

### Some Details
This project experimented with feature generation for image clustering. Using the classic MNIST dataset, I chose to focus on images of the numbers 5, 6, 8, and 9. Due to their similar structures, I thought these numbers would present more of a challenge to a clustering algorithm. I explored two methods.

Feature extraction with a pre-trained CNN:

- *cluster_transferlearning_featextraction.ipynb*
>
>After isolating the numbers 5, 6, 8, and 9 from the full dataset, I sent the images through a pre-trained CNN and saved the output after the final convolutional layer. Using this output as the set of features, I applied a k-means algorithm to cluster the images. Plotting a histogram of the image labels present in each cluster revealed a mix of numbers in each group. Ideally, each cluster would be pure, containing only images of a single number.

Feature generation with autoencoders:
- *model.py* and *config.py*
>I defined the autoencoder structure and training parameters to encode and decode the images. I used the same model structure to train an autoencoder for each number, producing four trained autoencoders.

- *pytorch-cnn-autoencoder_trainingrun.ipynb*

>To show an example of training, this notebook trains an autoencoder for the number '5' and displays the training progress. As the training advances through each epoch, the loss decreases and the image is more faithfully reconstructed.

- *cluster_autoencoder_featgeneration.ipynb*

>Since I saved the state dictionary for each trained autoencoder, I ran a test dataset through each pre-trained autoencoder to get loss values for each image. The loss values produced by an individual autoencoder indicated how well it reconstructed the original image. The images that most closely match the autoencoder's training data would, therefore, have the lowest loss values, and images different from the training set would have higher loss values. By saving the loss values produced by each autoencoder, I created a set of features to use for clustering.

- *cluster-autoencoderfeats.ipynb*

>Using the loss values produced by each autoencoder as features, I applied k-means clustering to group the images. This resulted in fairly pure clusters, shown by histograms displaying the image labels present in each cluster.

The structural similarity of the numbers 5, 6, 8, and 9 did seem to challenge k-means clustering. Extracting features with a pre-trained CNN yielded mixed clusters with multiple numbers per cluster. Training autoencoders on each number and using the autoencoder loss scores of the test dataset as features yielded much cleaner clusters. With a different dataset, the clustering results of the two methods could be different.

### Language
Python

### Packages Used
pytorch, sklearn, numpy, matplotlib

### Models
convolutional neural network (CNN), autoencoder, K-means

### Resources
[PyTorch documentation](https://pytorch.org/docs/stable/index.html)

