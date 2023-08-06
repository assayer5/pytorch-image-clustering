## Project: Image Clustering

### Overview
Clustered images of numbers 5, 6, 8, and 9 from MNIST dataset. Tested two methods: feature extraction from pre-trained model, feature generation with autoencoders.

Since the numbers 5, 6, 8, and 9 have similar structures, extracting features with a pre-trained model and clustering yielded mixed clusters with multiple numbers per cluster. Training autoencoders on each number and using the autoencoder loss scores of the test dataset as features yielded much cleaner clusters. With a different dataset, results could be different.

### Language
Python

### Packages Used
pytorch, sklearn, numpy, matplotlib

### Models
convolutional neural network (CNN), autoencoder, K-means

### Resources
[PyTorch documentation](https://pytorch.org/docs/stable/index.html)

