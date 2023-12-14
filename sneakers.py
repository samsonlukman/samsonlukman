# Set environment variable to suppress TensorFlow GPU warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

# Load and preprocess the image dataset
dataset = keras.preprocessing.image_dataset_from_directory(
    directory="imag", label_mode=None, image_size=(64, 64), batch_size=32,
    shuffle=True
).map(lambda x: x/255.0)

# Define the discriminator model
discriminator = keras.Sequential(
    [
        keras.Input(shape=(64,64,3)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Display discriminator summary
print(discriminator.summary())

# Define the generator model
latent_dim = 512
generator = keras.Sequential(
    [
        layers.Input(shape=(latent_dim,)),
        layers.Dense(8*8*128),
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
    ]
)

# Display generator summary
generator.summary()

# Define optimizers and loss function
opt_gen = keras.optimizers.Adam(1e-4)
opt_disc = keras.optimizers.Adam(1e-4)
loss_fn = keras.losses.BinaryCrossentropy()

# Set parameters for generating images
num_images_per_epoch = 2
output_folder = r'C:\\Users\\dell\\Pictures\\generated_images'
os.makedirs(output_folder, exist_ok=True)

# Training loop
for epoch in range(100):
    for idx, real in enumerate(tqdm(dataset)):
        batch_size = real.shape[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        fake = generator(random_latent_vectors)

        # Generate image at every even index
    # Save a set number of generated images at the end of each epoch
    for i in range(num_images_per_epoch):
        generated_image_path = os.path.join(output_folder, f'generated_image_epoch_{epoch}_iter_{i}.png')
        plt.imsave(generated_image_path, fake[i].numpy())


        # Train Discriminator: max log(D(x)) + log(1 - D(G(z))
        with tf.GradientTape() as disc_tape:
            loss_disc_real = loss_fn(tf.ones((batch_size, 1)), discriminator(real))
            loss_disc_fake = loss_fn(tf.zeros(batch_size, 1), discriminator(fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

        # Update discriminator weights
        grads = disc_tape.gradient(loss_disc, discriminator.trainable_weights)
        opt_disc.apply_gradients(
            zip(grads, discriminator.trainable_weights)
        )

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        with tf.GradientTape() as gen_tape:
            fake = generator(random_latent_vectors)
            output = discriminator(fake)
            loss_gen = loss_fn(tf.ones(batch_size, 1), output)

        # Update generator weights
        grads = gen_tape.gradient(loss_gen, generator.trainable_weights)
        opt_gen.apply_gradients(
            zip(grads, generator.trainable_weights)
        )
