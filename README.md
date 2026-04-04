Using a CNN and a dataset from Kaggle containing chest X-ray images of patients with and without pneumonia, I trained a model that predicts whether patients have pneumonia
with roughly 89% test accuracy. Additionally, it correctly predicted 97% of the test set's pneumonia images. It did struggle a bit with predicting the non-pneumonia patients, but this
error type isn't typically too big of a deal in this context. And this is probably partly due to the training set being roughly 74% pneumonia images. It could probably
be fixed somewhat by just including more non-pneumonia images in the training set. Also, the test set was better balanced as roughly 63% of the test images were of patients
with pneumonia.
