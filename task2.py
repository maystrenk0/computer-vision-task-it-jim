import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten, Dropout, Dense
from tensorflow.keras import Sequential, Model

import os
import random
import tensorflow as tf


seed = 42
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
(train_imgs, train_labels), (test_imgs, test_labels) = tf.keras.datasets.mnist.load_data()

train_imgs, test_imgs = train_imgs / 255.0, test_imgs / 255.0

print("Size of train images: {}, Number of train images: {}".format(train_imgs.shape[-2:], train_imgs.shape[0]))
print("Size of test images: {}, Number of test images: {}".format(test_imgs.shape[-2:], test_imgs.shape[0]))

plt.imshow(train_imgs[1], cmap='Greys')
plt.show()

plt.imshow(test_imgs[0], cmap='Greys')
plt.show()
plt.close()


def lrelu(x, alpha=0.01):
    return tf.math.maximum(alpha * x, x)


encoder = Sequential([
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        use_bias=True,
        activation=lrelu,
        name='conv1'
    ),
    MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        name='pool1'
    ),
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        use_bias=True,
        activation=lrelu,
        name='conv2'
    ),
    MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        name='encoding'
    )
])

decoder = Sequential([
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        name='conv3',
        padding='SAME',
        use_bias=True,
        activation=lrelu
    ),
    Conv2DTranspose(
        filters=32,
        kernel_size=3,
        padding='same',
        strides=2,
        name='upsample1'
    ),
    Conv2DTranspose(
        filters=32,
        kernel_size=3,
        padding='same',
        strides=2,
        name='upsample2'
    ),
    Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        name='logits',
        padding='SAME',
        use_bias=True
    )
])


class EncoderDecoderModel(Model):
    def __init__(self, is_sigmoid=False):
        super(EncoderDecoderModel, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._is_sigmoid = is_sigmoid

    def call(self, x):
        x = self._encoder(x)
        decoded = self._decoder(x)
        if self._is_sigmoid:
            decoded = tf.keras.activations.sigmoid(decoded)
        return decoded

    def save_model(self):
        self._encoder.save('encoder.h5')
        self._decoder.save('decoder.h5')


def add_noise(input_imgs, noise_factor=0.5):
    noisy_imgs = input_imgs + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=input_imgs.shape)
    noisy_imgs = np.clip(noisy_imgs, 0., 1.)
    return noisy_imgs


train_imgs_data = train_imgs[..., tf.newaxis]
test_imgs_data = test_imgs[..., tf.newaxis]

train_noisy_imgs = add_noise(train_imgs_data)
test_noisy_imgs = add_noise(test_imgs_data)

image_id_to_plot = 0
plt.imshow(tf.squeeze(train_noisy_imgs[image_id_to_plot]), cmap='Greys')
plt.title("The number is: {}".format(train_labels[image_id_to_plot]))
plt.show()

plt.imshow(tf.squeeze(test_noisy_imgs[image_id_to_plot]), cmap='Greys')
plt.title("The number is: {}".format(test_labels[image_id_to_plot]))
plt.show()
plt.close()


def cost_function(labels=None, logits=None, name=None):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name=name)
    return tf.reduce_mean(loss)


encoder_decoder_model = EncoderDecoderModel()

num_epochs = 25
batch_size_to_set = 64

learning_rate = 1e-5
num_workers = 2

encoder_decoder_model.compile(
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
    loss=cost_function,
    metrics=None
)

results = encoder_decoder_model.fit(
    train_noisy_imgs,
    train_imgs_data,
    epochs=num_epochs,
    batch_size=batch_size_to_set,
    validation_data=(test_noisy_imgs, test_imgs_data),
    workers=num_workers,
    shuffle=True
)

encoder_decoder_model.save_model()

encoder_decoder_model2 = EncoderDecoderModel(is_sigmoid=True)

img_num_to_decode = 10

test_imgs_data_decode = test_imgs_data[:img_num_to_decode]

test_noisy_imgs_decode = tf.cast(test_noisy_imgs[:img_num_to_decode], tf.float32)

decoded_images = encoder_decoder_model2(test_noisy_imgs_decode)
plt.figure(figsize=(20, 4))
plt.title('Reconstructed Images')

print("Original Images")
for i in range(img_num_to_decode):
    plt.subplot(2, img_num_to_decode, i + 1)
    plt.imshow(test_imgs_data_decode[i, ..., 0], cmap='gray')
plt.show()
plt.figure(figsize=(20, 4))

print("Noisy Images")
for i in range(img_num_to_decode):
    plt.subplot(2, img_num_to_decode, i + 1)
    plt.imshow(test_noisy_imgs_decode[i, ..., 0], cmap='gray')
plt.show()
plt.figure(figsize=(20, 4))

print("Reconstruction of Noisy Images")
for i in range(img_num_to_decode):
    plt.subplot(2, img_num_to_decode, i + 1)
    plt.imshow(decoded_images[i, ..., 0], cmap='gray')
plt.show()
plt.close()

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

clf = Sequential(
    [
        Input(shape=(28, 28, 1)),
        Conv2D(32, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ]
)

clf.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

clf.fit(train_imgs_data, train_labels, batch_size=128, epochs=15, validation_split=0.1)

score_clean = clf.evaluate(test_imgs_data, test_labels, verbose=0)
print("CLEAN - Test loss:", score_clean[0])
print("CLEAN - Test accuracy:", score_clean[1])
score_noisy = clf.evaluate(test_noisy_imgs, test_labels, verbose=0)
print("NOISY - Test loss:", score_noisy[0])
print("NOISY - Test accuracy:", score_noisy[1])
decoded_imgs = encoder_decoder_model2(tf.cast(test_noisy_imgs, tf.float32))
score_denoised = clf.evaluate(decoded_imgs, test_labels, verbose=0)
print("DENOISED - Test loss:", score_denoised[0])
print("DENOISED - Test accuracy:", score_denoised[1])
