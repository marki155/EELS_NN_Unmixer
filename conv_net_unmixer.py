import os
import tensorflow as tf

import mod_hyperspy.hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from eels_fit_encoding import parameters_to_model, model_to_parameters, parameters_to_I



def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

class ActivityRegularization(tf.keras.layers.Layer):
    def __init__(self, rate=1e-5):
        super(ActivityRegularization, self).__init__()
        self.rate = rate

    def call(self, inputs):
        #self.add_loss(self.rate * tf.reduce_sum(tf.abs(inputs)))
        denominator = tf.reduce_sum(inputs, axis=-1)
        length = tf.reduce_sum(tf.ones_like(denominator))
        denominator = tf.reshape(tf.repeat(denominator, inputs.shape[1]), [length, inputs.shape[1]])
        out = inputs / denominator
        out = tf.where(tf.math.is_nan(out), tf.zeros_like(out), out)
        self.add_loss((tf.reduce_sum(out) - tf.reduce_max(out)) * self.rate)
        return out

class SparsityEnhancingLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(SparsityEnhancingLayer, self).__init__(**kwargs)
        self.units = units
        if activation is None:
            self.act_fkt = tf.math.sigmoid
        else:
            self.act_fkt = activation

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha',
                                      shape=(self.units,),
                                      initializer='zeros',
                                      trainable=True)
        super(SparsityEnhancingLayer, self).build(input_shape)

    def call(self, inputs):
        theta_alpha = self.act_fkt(inputs - self.alpha)
        return theta_alpha

    def get_config(self):
        config = super(SparsityEnhancingLayer, self).get_config()
        config.update({'units': self.units})
        return config


class DecoderDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, E_size, rate=1.0, **kwargs):
        super(DecoderDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.E_size = E_size
        #self.endmembers_hist = []
        self.rate = rate

    def build(self, input_shape):
        self.endmembers = self.add_weight(name="decoder_endmembers",
                                 shape=(self.E_size, self.units),
                                 initializer=tf.keras.initializers.GlorotUniform(seed=0),
                                 trainable=True)
        self.noise = self.add_weight(name="decoder_bias",
                                 shape=(self.E_size,),
                                 initializer='zeros',
                                 trainable=True)
        super(DecoderDenseLayer, self).build(input_shape)

    def call(self, inputs):
        #self.endmembers_hist.append(self.endmembers)
        negatives = tf.boolean_mask(self.endmembers, self.endmembers < 0)
        sum_negatives = tf.reduce_sum(negatives)
        self.add_loss(tf.abs(sum_negatives) * self.rate)
        return tf.matmul(inputs, tf.transpose(self.endmembers)) + self.noise




#def spectral_angle_distance(x, y):
#    x_norm = tf.norm(x, axis=-1, keepdims=True)
#    y_norm = tf.norm(y, axis=-1, keepdims=True)###
#
    # Berechnung des Skalarprodukts zwischen den Vektoren
#    dot_product = tf.reduce_sum(x * y, axis=-1, keepdims=True)

    # Berechnung des Kosinus des Winkels
 #   cos_angle = dot_product / (x_norm * y_norm)
#    cos_angle = tf.clip_by_value(cos_angle, -0.9999, 0.9999)
#
    # Berechnung des Winkels
#    angle = tf.math.acos(cos_angle)

    # Rückgabe des Ergebnisses
#    return angle

def spectral_angle_distance(y_true, y_pred):
    # Berechnung der Länge der Vektoren
    x_norm = tf.norm(y_true, axis=-1)
    y_norm = tf.norm(y_pred, axis=-1)



    # Berechnung des Skalarprodukts zwischen den Vektoren
    dot_product = tf.reduce_sum(y_true * y_pred, axis=-1)


    # Berechnung des Kosinus des Winkels
    cos_angle = dot_product / (x_norm * y_norm)

    # Berechnung des Winkels
    angle = tf.math.acos(cos_angle)
    return tf.reduce_mean(angle)# + tf.losses.MSE(y_true, y_pred)

class ConvNetUnmixer:
    def __init__(self, fusion_data, fusion_data_binned, save_name, E_cut=None, empty=False):
        if not empty:
            if E_cut is not None:
                E_max = E_cut
            else:
                E_max = fusion_data.shape[1]
            self.n_pixel = fusion_data.shape[0]
            self.n_pixel_train = fusion_data_binned.shape[0]

            self.train_data = fusion_data_binned[:, :E_max].astype('float32')
            self.val_data = fusion_data[:, :E_max].astype('float32')
            self.E_max = E_max

        self.checkpoint_path = save_name
        if self.checkpoint_path is not None:
            self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
            self.latest = tf.train.latest_checkpoint(self.checkpoint_dir)


    def sample_spectrum_li(self):

        sample = hs.load('Li Test\\EELS Spectrum Image (high-loss).dm4')
        ll = hs.load('Li Test\\EELS Spectrum Image (low-loss).dm4')
        #sample = remove_eels_background_with_fitting(sample, ll)

        eds = hs.load('Li Test\\EDS Spectrum Image.dm4')
        ll.align_zero_loss_peak(also_align=[sample])

        self.train_data, self.n_pixel_train, self.E_max = self.prepare_data(ll, sample, eds, binning_factor=4)

        self.val_data, self.n_pixel, E_max = self.prepare_data(ll, sample, eds)

        self.o_x, self.o_y, _ = sample.data.shape


    def sample_spectrum_si_pore(self):

        sample = hs.load('Si Pore\\EELS Spectrum Image (high-loss) (aligned).dm4')
        ll = hs.load('Si Pore\\EELS Spectrum Image (low-loss) (aligned).dm4')
        #sample = remove_eels_background_with_fitting(sample, ll)

        eds = hs.load('Si Pore\\EDS Spectrum Image.dm4')
        ll.align_zero_loss_peak(also_align=[sample])

        self.train_data, self.n_pixel_train, self.E_max = self.prepare_data(ll, sample, eds, binning_factor=4)

        self.val_data, self.n_pixel, E_max = self.prepare_data(ll, sample, eds)

        self.o_x, self.o_y, _ = sample.data.shape




    def prepare_data(self, ll_eels, cl_eels, eds, binning_factor=None):
        from test import make_datafusion
        n_x, n_y, _ = cl_eels.data.shape
        if binning_factor is not None:
            b_x = (n_x // binning_factor) * binning_factor
            b_y = (n_y // binning_factor) * binning_factor
            s_binning = (b_x//binning_factor, b_y//binning_factor)
        else:
            s_binning = None
            b_x = n_x
            b_y = n_y
        fusion_data, splits_i, kron_vec = make_datafusion([cl_eels.data[:b_x,:b_y,:]],
                                                          [1],
                                                          KRON=False, sample_binning=s_binning)
        #fusion_data, splits_i, kron_vec = make_datafusion([cl_eels.data[:b_x, :b_y, :]],
        #    [1],
        #    KRON=False, sample_binning=s_binning)

        #fusion_data = fusion_data / np.max(fusion_data)


        E_max = fusion_data.shape[1]
        n_pixel = fusion_data.shape[0]

        train_data = fusion_data[:n_pixel, :E_max].astype('float32')

        return train_data, n_pixel, E_max


    def build_model(self, R, g='sigmoid', end_act=None, learning_rate=0.0001, loss_fkt=spectral_angle_distance,
                    dropout_rate=0.1, neg_rate=1.0, ab_rate=1e-5, layer_units=None):


        input = tf.keras.layers.Input(shape=(self.E_max), name="encoder_input")

        if layer_units is None:
            layer_units = [9, 6, 3, 1]

        encoder = input

        for i, unit in enumerate(layer_units):
            encoder = tf.keras.layers.Dense(units=unit * R, name="encoder_dense_" + str(i), activation=g,
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=12))(encoder)

        encoder = tf.keras.layers.BatchNormalization()(encoder)

        encoder = SparsityEnhancingLayer(R, activation=end_act)(encoder) # Dynamical Soft Thresholding

        encoder = ActivityRegularization(rate=ab_rate)(encoder)

        encoder = tf.keras.layers.GaussianDropout(dropout_rate)(encoder)


        output = DecoderDenseLayer(R, self.E_max, rate=neg_rate)(encoder)



        self.autoencoder = tf.keras.models.Model(input, output, name="AE")




        self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                 run_eagerly=False, loss=loss_fkt, metrics=['mean_squared_error'])
        self.autoencoder.summary()
        return self.autoencoder



    def load_last_training(self):
        if self.latest is None:
            if self.checkpoint_path is not None:
                self.autoencoder.load_weights(self.checkpoint_path)
        else:
            self.autoencoder.load_weights(self.latest)



    def training(self, epochs, batch_size, all_callbacks):
        STEPS_PER_EPOCH = self.n_pixel_train / batch_size
        SAVE_PERIOD = 100

        # Create a callback that saves the model's weights
        if self.checkpoint_path is not None:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1, save_freq=int(SAVE_PERIOD * STEPS_PER_EPOCH))
        else:
            cp_callback = None

        filename = 'log.csv'
        history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

        if not cp_callback is None:
            all_callbacks.append(cp_callback)


        history = self.autoencoder.fit(self.train_data, self.train_data, epochs=epochs, batch_size=batch_size,
                                       steps_per_epoch=STEPS_PER_EPOCH, callbacks=all_callbacks, verbose=2)



        (loss, err) = self.autoencoder.evaluate(self.val_data, self.val_data)
        print("[INFO] test mean square error: {:.4f}".format(err))

        return loss, err

    def loss(self):
        return self.autoencoder.evaluate(self.train_data, self.train_data)

    def get_endmembers(self):
        return tf.transpose(tf.constant(self.autoencoder.layers[-1].get_weights()[0], dtype='float32'))

    def get_maps(self):
        encoder = tf.keras.models.Sequential()
        for i, layer in enumerate(self.autoencoder.layers[:-2]):
            encoder.add(layer)
        encoded = encoder(self.val_data)

        data = encoded.numpy()


        return data

    def calc_endmember_distances(self):
        H = self.get_endmembers()
        R = H.shape[0]
        mat = np.zeros((R, R))
        for i in range(R):
            for j in range(R):
                mat[i,j] = np.mean((H[i,:] - H[j,:])**2)
        return np.mean(mat)

    def get_error(self):
        loss, SAD = self.autoencoder.evaluate(self.val_data, self.val_data)
        return loss, SAD

    def evaluation(self):
        endmembers = self.get_endmembers()

        maps = self.get_maps()
        R = endmembers.shape[0]

        fig, axs = plt.subplots(R,2)

        xx, yy = (self.o_x, self.o_y)


        for r in range(R):
            axs[r, 1].plot(list(range(self.E_max)), endmembers[r, :])
            axs[r, 0].imshow(np.reshape(maps[:, r], (xx, yy)))

        plt.show()

    def stepwise_training(self, R, rounds=None):
        if rounds is None:
            rounds = [(500,10, 0.001, 0.01), (1000,3, 0.001, 0.2)]

        for epochs, batch_s, learning_rate, dropout in rounds:
            self.build_model(R=R, learning_rate=learning_rate, loss_fkt=spectral_angle_distance, dropout_rate=dropout)
            print("current loss:" + str(self.loss()))
            self.training(epochs=epochs, batch_size=batch_s)
            self.load_last_training()

    def __del__(self):
        pass
        #tf.keras.backend.clear_session()


if __name__ == "__main__":
    unmixer = ConvNetUnmixer(None, None, "./checkpoints/Si_pore.ckpt", E_cut=None, empty=True)
    unmixer.sample_spectrum_si_pore()
    R = 4


    if True:
        unmixer.stepwise_training(R)
    else:
        unmixer.build_model(R=R, learning_rate=0.001, loss_fkt=spectral_angle_distance, dropout_rate=0.01)
        unmixer.load_last_training()
    unmixer.evaluation()