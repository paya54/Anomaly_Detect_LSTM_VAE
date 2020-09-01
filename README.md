# Anomaly detection based on LSTM Variational AutoEncoder (LSTM-VAE)

# Description
The code in this repo shows how to construct LSTM-VAE model to detect anomalies based on [this paper](https://arxiv.org/pdf/1711.00614). Similar to LSTM AE model, LSTM-VAE is also a reconstruction-based anomaly detection model, which consists of a pair of encoder and decoder. 

The encoder is comprised of a LSTM network and two linear neural networks to estimate the mean and co-variance of the latent variable z.

The decoder has similar structure with a LSTM network and two linear neural networks to estimate the mean and co-variance of the reconstructed variable x_hat.

The anomaly detection is based on so-called anomaly score, which is defined as the log-likelihood of an input observation respect with the reconstructed mean and co-variance.

# Data 
The code uses [NASA bearing data set](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) for training and test. The bearing data has been uploaded to Azure Machine Learning workspace and registered as dataset. You can download the raw data and train the model in your local machine.

# Dependencies
This code has been tested with Python: 3.7.7, Tensorflow 2.2.0 and Keras 2.3.0