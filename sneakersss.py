import os
import random
from tqdm import tqdm
from typing import List
import shutil
import pandas as pd
import zipfile
from tensorflow import keras
import numpy as np
import glob
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Reshape, Embedding, concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras import datasets, layers, models
import cv2
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

print("Starting...")

# Specify the target directory for extraction
base_directory = r'C:\\Users\\dell\\Pictures\\sneakers_zip'

try:
    image_folder = 'images'
    csv_file = 'styles.csv'
    sneakers_folder = r'C:\\Users\\dell\\Pictures\\sneakers'

    # Create the sneakers folder if it doesn't exist
    if not os.path.exists(sneakers_folder):
        os.mkdir(sneakers_folder)

    try:
        # Read the styles.csv file using pandas and skip bad lines
        df = pd.read_csv('styles.csv', on_bad_lines='skip')
        print("CSV file read successfully")
    except pd.errors.ParserError as ex:
        print(f"Error parsing CSV file: {ex}")
        df = pd.read_csv(csv_file, on_bad_lines='skip')

    # Filter images with 'Casual Shoes' or 'Sport Shoes' in the 'articleType' column
    sneaker_images = df[df['articleType'].isin(['Casual Shoes', 'Sports Shoes'])]
    print("Sneaker_images filtered")
    # Copy selected images to the 'sneakers' folder
    for index, row in sneaker_images.iterrows():
        image_id = str(row['id']) + ".jpg"  # Assuming the image filenames include ".jpg"
        source_path = os.path.join(image_folder, image_id)
        destination_path = os.path.join(sneakers_folder, image_id)

        # Check if the image file exists before copying
        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)

    print("Sneaker images copied to the 'sneakers' folder.")
except Exception as ex:
    print(f"An error occurred: {ex}")

sneakers = r'C:\\Users\\dell\\Pictures\\sneakers'
image_files = glob.glob(os.path.join(sneakers, '*.jpg'))

# Get the count of image files
num_images = len(image_files)

print(f'This is the total number of images in the folder: {num_images}')

# Load styles data
styles_df = pd.read_csv('styles.csv', on_bad_lines='skip')

# Map gender labels to integers
gender_mapping = {'Boys': 1, 'Girls': 2, 'Women': 3, 'Men': 4, 'Unisex': 5}
styles_df['gender'] = styles_df['gender'].map(gender_mapping)

# Define hyperparameters
z_dim = 100
label_dim = len(gender_mapping)
image_shape = (128, 128, 3)

# Load and preprocess images
sneakers_folder = r'C:\\Users\\dell\\Pictures\\sneakers'
image_width = 128
image_height = 128

images = []
labels = []

for root, _, files in os.walk(sneakers_folder):
    for file in files:
        image_path = os.path.join(root, file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_width, image_height))
        image = image / 255.0
        label = int(os.path.splitext(os.path.basename(file))[0])
        images.append(image)
        labels.append(label)

# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Visualize the data (example for the first 9 images)
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    ax.set_title(f'Label: {labels[i]}')
plt.show()

# Calculate the frequency of each label in the training set
label_counts = styles_df['gender'].value_counts().reset_index()
label_counts.columns = ['Gender', 'Frequency']

# Create a bar chart using Seaborn
plt.figure(figsize=(8, 6))
sns.barplot(x='Gender', y='Frequency', data=label_counts)

# Add labels and title
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title('Gender Distribution in the Styles Dataset')

# Show the plot
plt.xticks(rotation=45)
plt.show()


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print(labels)

z_dim = 100  # Dimension of the random noise vector
label_dim = 6  # Number of unique labels

image_shape = (128, 128, 3)  # Adjust to match your image dimensions

def build_generator(z_dim, label_dim, image_shape):
    # Input for random noise vector
    z = Input(shape=(z_dim,))
    # Input for label
    label = Input(shape=(1,), dtype='int32')

    # Embedding layer for labels
    label_embedding = Flatten()(Embedding(label_dim, z_dim)(label))

    # Concatenate noise and label inputs
    combined = concatenate([z, label_embedding])

    # Fully connected layers to generate an image
    x = Dense(128, input_dim=z_dim + label_dim)(combined)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = Dense(np.prod(image_shape), activation='tanh')(x)
    x = Reshape(image_shape)(x)

    return Model([z, label], x)

def build_discriminator(image_shape, label_dim):
    # Input for the image
    img = Input(shape=image_shape)
    # Input for label
    label = Input(shape=(1,), dtype='int32')

    # Embedding layer for labels
    label_embedding = Flatten()(Embedding(label_dim, np.prod(image_shape))(label))

    # Flatten the image
    img_flatten = Flatten()(img)

    # Concatenate image and label inputs
    combined = concatenate([img_flatten, label_embedding])

    # Fully connected layers
    x = Dense(128)(combined)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model([img, label], x)

styles_df = pd.read_csv('styles.csv', on_bad_lines='skip')

# Use the 'gender' column from the dataset as labels
labels = styles_df['gender']
# Map string labels to integers
label_mapping = {'Boys': 1, 'Girls': 2, 'Women': 3, 'Men': 4, 'Unisex': 5}
encoded_labels = labels.map(label_mapping)

# Check encoded_labels
print(encoded_labels)

# Define hyperparameters
z_dim = 100  # Dimension of the random noise vector
label_dim = 6  # Number of unique labels
image_shape = (128, 128, 3)

# Check X_train shape
print("X_train shape:", X_train.shape)

# Build the generator
generator = build_generator(z_dim, label_dim, image_shape)

# Build the discriminator
discriminator = build_discriminator(image_shape, label_dim)

# GAN input layers
gan_input_z = Input(shape=(z_dim,))
gan_input_label = Input(shape=(1,), dtype='int32')  # Ensure labels are of integer type

# Generate images from the generator
generated_image = generator([gan_input_z, gan_input_label])

# Ensure that the discriminator is not trainable during GAN training
discriminator.trainable = False

# Get the validity of generated images
validity = discriminator([generated_image, gan_input_label])

# Create the GAN model
gan = Model([gan_input_z, gan_input_label], validity)

# Compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

# Compile the GAN
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0006, beta_1=0.5))

opt_gen = keras.optimizers.Adam(1e-5)
opt_disc = keras.optimizers.Adam(1e-5)
loss_fn = keras.losses.BinaryCrossentropy()
# Define the checkpoint directory and manager
checkpoint_dir = r'C:\\Users\\dell\\Pictures\\generated_images'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=opt_gen,
                                 discriminator_optimizer=opt_disc,
                                 generator=generator,
                                 discriminator=discriminator)

# Restore the latest checkpoint if available
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print(f"Restored from {checkpoint_manager.latest_checkpoint}")
else:
    print("Initializing from scratch.")
# Define batch size and number of epochs
import numpy as np

# Define the batch size and number of epochs
batch_size = 256
epochs = 1500
display_interval = 100

# Define the number of images you want to generate
num_images_to_generate = 10
for epoch in range(epochs):
    # Determine the current batch size based on the available data
    current_batch_size = min(batch_size, X_train.shape[0])

    # Generate random noise vectors for the generator
    noise = np.random.normal(0, 1, size=(current_batch_size, z_dim))

    # Generate random labels for the generated images
    generated_labels = np.random.choice(label_dim, current_batch_size)

    # Generate fake images from the generator
    generated_images = generator.predict([noise, generated_labels])

    # Select a random batch of real images
    indices = np.random.choice(X_train.shape[0], current_batch_size, replace=False)
    real_images = X_train[indices]
    real_labels = encoded_labels[indices]

    # Create target labels for the discriminator
    valid = np.ones((current_batch_size, 1))
    fake = np.zeros((current_batch_size, 1))
    d_loss_real = discriminator.train_on_batch([real_images, real_labels], valid)
    d_loss_fake = discriminator.train_on_batch([generated_images, generated_labels], fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the GAN (generator) by trying to generate valid images
    g_loss = gan.train_on_batch([noise, generated_labels], valid)

    # Print progress
    print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

    # Display or save generated images at specified intervals
    if epoch % display_interval == 0:
        # Generate random noise vectors for the generator
        noise = np.random.normal(0, 1, size=(num_images_to_generate, z_dim))

        # Generate random labels for the generated images
        generated_labels = np.random.choice(label_dim, num_images_to_generate)

        # Generate fake images from the generator
        generated_images = generator.predict([noise, generated_labels])

        # Visualize the generated images
        fig, axes = plt.subplots(1, num_images_to_generate, figsize=(15, 15))
        for i in range(num_images_to_generate):
            # Convert the image from the range [-1, 1] to [0, 1]
            image_to_show = (generated_images[i] + 1) / 2.0
            axes[i].imshow(image_to_show)
            axes[i].axis('off')
        plt.show()

        # Save the generated images
        output_folder = r'C:\\Users\\dell\\Pictures\\generated_images'
        os.makedirs(output_folder, exist_ok=True)

        for i in range(num_images_to_generate):
            generated_image_path = os.path.join(output_folder, f'generated_image_epoch_{epoch}_iter_{i}.png')

            # Convert the image from the range [-1, 1] to [0, 255]
            image_to_save = ((generated_images[i] + 1) * 127.5).astype(np.uint8)

            # Use OpenCV to save the image
            cv2.imwrite(generated_image_path, cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR))

            print(f"Generated image saved: {generated_image_path}")