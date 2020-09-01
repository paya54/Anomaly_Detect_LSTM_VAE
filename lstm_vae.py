from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace, Dataset
import numpy as np
np.random.seed(0)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
tf.random.set_seed(0)
from tensorflow import keras, data
import tensorflow_probability as tfp
from tensorflow.keras import layers, regularizers, activations
from tensorflow.keras import backend as K
import seaborn as sns
import matplotlib.pyplot as plt

ws = Workspace.from_config()

dataset_name = "bearing_dataset"
#train_ratio = 0.75
row_mark = 740
batch_size = 10
time_step = 1
x_dim = 4
lstm_h_dim = 8
z_dim = 4
epoch_num = 100
threshold = 0.03

mode = 'train'
model_dir = "./lstm_vae_model/"
image_dir = "./lstm_vae_images/"

def split_normalize_data(all_df):
    #row_mark = int(all_df.shape[0] * train_ratio)
    train_df = all_df[:row_mark]
    test_df = all_df[row_mark:]

    scaler = MinMaxScaler()
    scaler.fit(np.array(all_df)[:, 1:])
    train_scaled = scaler.transform(np.array(train_df)[:, 1:])
    test_scaled = scaler.transform(np.array(test_df)[:, 1:])
    return train_scaled, test_scaled

def reshape(da):
    return da.reshape(da.shape[0], time_step, da.shape[1]).astype("float32")

class Sampling(layers.Layer):
    def __init__(self, name='sampling_z'):
        super(Sampling, self).__init__(name=name)
    
    def call(self, inputs):
        mu, logvar = inputs
        print('mu: ', mu)
        sigma = K.exp(logvar * 0.5)
        epsilon = K.random_normal(shape=(mu.shape[0], z_dim), mean=0.0, stddev=1.0)
        return mu + epsilon * sigma
    
    def get_config(self):
        config = super(Sampling, self).get_config()
        config.update({'name': self.name})
        return config

class Encoder(layers.Layer):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.encoder_inputs = keras.Input(shape=(time_step, x_dim))
        self.encoder_lstm = layers.LSTM(lstm_h_dim, activation='softplus', name='encoder_lstm', stateful=True)
        self.z_mean = layers.Dense(z_dim, name='z_mean')
        self.z_logvar = layers.Dense(z_dim, name='z_log_var')
        self.z_sample = Sampling()
    
    def call(self, inputs):
        self.encoder_inputs = inputs
        hidden = self.encoder_lstm(self.encoder_inputs)
        mu_z = self.z_mean(hidden)
        logvar_z = self.z_logvar(hidden)
        z = self.z_sample((mu_z, logvar_z))
        return mu_z, logvar_z, z
    
    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'name': self.name,
            'z_sample': self.z_sample.get_config()
        })
        return config

class Decoder(layers.Layer):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.z_inputs = layers.RepeatVector(time_step, name='repeat_vector')
        self.decoder_lstm_hidden = layers.LSTM(lstm_h_dim, activation='softplus', return_sequences=True, name='decoder_lstm')
        self.x_mean = layers.Dense(x_dim, name='x_mean')
        self.x_sigma = layers.Dense(x_dim, name='x_sigma', activation='tanh')
    
    def call(self, inputs):
        z = self.z_inputs(inputs)
        hidden = self.decoder_lstm_hidden(z)
        mu_x = self.x_mean(hidden)
        sigma_x = self.x_sigma(hidden)
        return mu_x, sigma_x
    
    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'name': self.name
        })
        return config

loss_metric = keras.metrics.Mean(name='loss')
likelihood_metric = keras.metrics.Mean(name='log likelihood')

class LSTM_VAE(keras.Model):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='lstm_vae', **kwargs):
        super(LSTM_VAE, self).__init__(name=name, **kwargs)

        self.encoder = Encoder(time_step, x_dim, lstm_h_dim, z_dim, **kwargs)
        self.decoder = Decoder(time_step, x_dim, lstm_h_dim, z_dim, **kwargs)
    
    def call(self, inputs):
        mu_z, logvar_z, z = self.encoder(inputs)
        mu_x, sigma_x = self.decoder(z)

        var_z = K.exp(logvar_z)
        kl_loss = K.mean(-0.5 * K.sum(var_z - logvar_z + tf.square(1 - mu_z), axis=1), axis=0)
        self.add_loss(kl_loss)

        dist = tfp.distributions.Normal(loc=mu_x, scale=tf.abs(sigma_x))
        log_px = -dist.log_prob(inputs)

        return mu_x, sigma_x, log_px
    
    def get_config(self):
        config = {
            'encoder': self.encoder.get_config(),
            'decoder': self.decoder.get_config(),
            'name': self.name
        }
        return config
    
    def reconstruct_loss(self, x, mu_x, sigma_x):
        var_x = K.square(sigma_x)
        reconst_loss = -0.5 * K.sum(K.log(var_x), axis=2) + K.sum(K.square(x - mu_x) / var_x, axis=2)
        reconst_loss = K.reshape(reconst_loss, shape=(x.shape[0], 1))
        return K.mean(reconst_loss, axis=0)

    def mean_log_likelihood(self, log_px):
        log_px = K.reshape(log_px, shape=(log_px.shape[0], log_px.shape[2]))
        mean_log_px = K.mean(log_px, axis=1)
        return K.mean(mean_log_px, axis=0)

    def train_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data
        
        with tf.GradientTape() as tape:
            mu_x, sigma_x, log_px = self(x, training=True)
            loss = self.reconstruct_loss(x, mu_x, sigma_x)
            loss += sum(self.losses)
            mean_log_px = self.mean_log_likelihood(log_px)
            
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        loss_metric.update_state(loss)
        likelihood_metric.update_state(mean_log_px)
        return {'loss': loss_metric.result(), 'log_likelihood': likelihood_metric.result()}

def plot_loss_moment(history):
    _, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history['loss'], 'blue', label='Loss', linewidth=1)
    ax.plot(history['log_likelihood'], 'red', label='Log likelihood', linewidth=1)
    ax.set_title('Loss and log likelihood over epochs')
    ax.set_ylabel('Loss and log likelihood')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.savefig(image_dir + 'loss_lstm_vae_' + mode + '.png')

def plot_log_likelihood(df_log_px):
    plt.figure(figsize=(14, 6), dpi=80)
    plt.title("Log likelihood")
    sns.set_color_codes()
    sns.distplot(df_log_px, bins=40, kde=True, rug=True, color='blue')
    plt.savefig(image_dir + 'log_likelihood_' + mode + '.png')

def save_model(model):
    with open(model_dir + 'lstm_vae.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights(model_dir + 'lstm_vae_ckpt')

def load_model():
    lstm_vae_obj = {'Encoder': Encoder, 'Decoder': Decoder, 'Sampling': Sampling}
    with keras.utils.custom_object_scope(lstm_vae_obj):
        with open(model_dir + 'lstm_vae.json', 'r'):
            model = keras.models.model_from_json(model_dir + 'lstm_vae.json')
        model.load_weights(model_dir + 'lstem_vae_ckpt')
    return model

def main():
    try:
        dataset = Dataset.get_by_name(ws, dataset_name)
        print("Dataset found: ", dataset_name)
    except Exception:
        print("Dataset not found: ", dataset_name)
    
    all_df = dataset.to_pandas_dataframe()
    train_scaled, test_scaled = split_normalize_data(all_df)
    x_dim = train_scaled.shape[1]
    print("train and test data shape after scaling: ", train_scaled.shape, test_scaled.shape)

    train_X = reshape(train_scaled)
    test_X = reshape(test_scaled)
    
    opt = keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-6, amsgrad=True)
    
    if mode == "train":
        model = LSTM_VAE(time_step, x_dim, lstm_h_dim, z_dim, dtype='float32')
        model.compile(optimizer=opt)
        train_dataset = data.Dataset.from_tensor_slices(train_X)
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

        history = model.fit(train_dataset, epochs=epoch_num, shuffle=False).history
        model.summary()
        plot_loss_moment(history)
        save_model(model)
    elif mode == "infer":
        model = load_model()
        model.compile(optimizer=opt)
    else:
        print("Unknown mode: ", mode)
        exit(1)
    
    _, _, train_log_px = model.predict(train_X, batch_size=1)
    train_log_px = train_log_px.reshape(train_log_px.shape[0], train_log_px.shape[2])
    df_train_log_px = pd.DataFrame()
    df_train_log_px['log_px'] = np.mean(train_log_px, axis=1)
    plot_log_likelihood(df_train_log_px)

    
    _, _, test_log_px = model.predict(test_X, batch_size=1)
    test_log_px = test_log_px.reshape(test_log_px.shape[0], test_log_px.shape[2])
    df_log_px = pd.DataFrame()
    df_log_px['log_px'] = np.mean(test_log_px, axis=1)
    df_log_px = pd.concat([df_train_log_px, df_log_px])
    df_log_px['threshold'] = 0.65
    df_log_px['anomaly'] = df_log_px['log_px'] > df_log_px['threshold']
    df_log_px.index = np.array(all_df)[:, 0]
    
    df_log_px.plot(logy=True, figsize=(16, 9), color=['blue', 'red'])
    plt.savefig(image_dir + 'anomaly_lstm_vae_' + mode + '.png')
    
if __name__ == "__main__":
    main()