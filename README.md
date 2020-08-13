# imageSimilarityUsingAutoencoderandKNN
In inference.py file provide the path required and then run it. It will give the output as input image and five similar image determined by the autoencoder and KNN model. Idea behind it is that we can use autoencoder to encode all the images and pass it to KNN model. KNN model Using cosine similarity will determine how similar/related the images are and it will provide top similar images based on cosine similarity score. This idea is taken from NLP where words are encoded using autoencoder or transformer and these encoded words are used to find the similar/relatated words using manhatten distance or cosine similarty. This technique suggest how far are these words from each other. Nearest the words are, more similar or realed they are.

mainTraining.ipynb file is used to train the autoencoder and KNN model

Autoencoder.py file contains structure of the autoencoder

DataGenerator file is used to create pipeline between data and the model
