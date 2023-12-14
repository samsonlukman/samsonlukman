import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define constants
IMG_SHAPE = (128, 128, 3)
LATENT_DIM = 100
BATCH_SIZE = 64
EPOCHS = 5000
SAVE_INTERVAL = 500

# Define paths
DATA_PATH = 'images'
SAVE_PATH = r'C:\\Users\\dell\\Pictures\\generated_images'

# Define Generator and Discriminator networks
def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(8 * 8 * 256, input_dim=latent_dim))
    model.add(layers.Reshape((8, 8, 256)))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(128, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(64, kernel_size=3, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(3, kernel_size=3, padding="same"))
    model.add(layers.Activation("tanh"))
    return model

def build_discriminator(img_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

# Build and compile the discriminator
discriminator = build_discriminator(IMG_SHAPE)
discriminator.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(0.0002, 0.5), metrics=["accuracy"])

# Build the generator
generator = build_generator(LATENT_DIM)

# Build and compile the combined model (stacked generator and discriminator)
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(LATENT_DIM,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = models.Model(gan_input, gan_output)
gan.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(0.0002, 0.5))

# Load and preprocess the dataset
def load_dataset(data_path, img_shape):
    images = []
    for filename in os.listdir(data_path):
        img = load_img(os.path.join(data_path, filename), target_size=img_shape[:2])
        img = img_to_array(img) / 127.5 - 1.0
        images.append(img)
    return np.array(images)

# Training loop
def train_gan(epochs, batch_size, save_interval):
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train discriminator with real images
        idx = np.random.randint(0, dataset.shape[0], batch_size)
        real_imgs = dataset[idx]
        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)

        # Train discriminator with fake images
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        fake_imgs = generator.predict(noise)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        g_loss = gan.train_on_batch(noise, real_labels)

        # Print progress
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

        # Save generated images at specified intervals
        if epoch % save_interval == 0:
            save_generated_images(epoch)

# Save generated images
def save_generated_images(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, LATENT_DIM))
    generated_imgs = generator.predict(noise) * 0.5 + 0.5
    generated_imgs = generated_imgs.reshape((r, c) + IMG_SHAPE)

    for i in range(r):
        for j in range(c):
            img = array_to_img(generated_imgs[i, j])
            img.save(os.path.join(SAVE_PATH, f"generated_sneaker_{epoch}_{i * c + j}.png"))

# Load dataset
dataset = load_dataset(DATA_PATH, IMG_SHAPE)

# Normalize the images to the range [-1, 1]
dataset = dataset / 127.5 - 1.0

# Train the GAN
train_gan(EPOCHS, BATCH_SIZE, SAVE_INTERVAL)
