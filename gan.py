import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

dataset = keras.preprocessing.image_dataset_from_directory(
    directory="imag", label_mode=None, image_size=(64, 64), batch_size=16,
    shuffle=True
).map(lambda x: x / 255.0)

normalized_dataset = dataset.map(lambda x: (x - 0.5) / 0.5)

discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
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

print(discriminator.summary())

latent_dim = 30
generator = keras.Sequential(
    [
        layers.Input(shape=(latent_dim,)),
        layers.Dense(8 * 8 * 256),  # Increased the dense layer size
        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(3, kernel_size=5, padding="same", activation="tanh"),
    ]
)

# Scale the output to [0, 255]
generator.add(layers.Lambda(lambda x: (x + 1.0) * 127.5, output_shape=(64, 64, 3)))

generator.summary()

# Add gradient clipping to the optimizers
opt_gen = keras.optimizers.Adam(1e-3, clipvalue=0.5)  # Adjusted learning rate
opt_disc = keras.optimizers.Adam(1e-3, clipvalue=0.5)  # Adjusted learning rate
loss_fn = keras.losses.BinaryCrossentropy()

checkpoint_dir = r'C:\\Users\\dell\\Pictures\\test_sneak'
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

num_images_per_epoch = 2
output_folder = r'C:\\Users\\dell\\Pictures\\test_sneak'
os.makedirs(output_folder, exist_ok=True)

for epoch in range(1000):
    for idx, real in enumerate(tqdm(dataset)):
        batch_size = real.shape[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        fake = generator(random_latent_vectors)

        # Save generated images at every iteration
        img = keras.preprocessing.image.array_to_img((fake[0] + 1.0) * 127.5)
        img.save(os.path.join(output_folder, f'generated_image_epoch_{epoch}_iter_{idx}.png'))

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z))
        with tf.GradientTape() as disc_tape:
            loss_disc_real = loss_fn(tf.random.uniform(shape=(batch_size, 1), minval=0.9, maxval=1.0),
                                      discriminator(real))
            loss_disc_fake = loss_fn(tf.random.uniform(shape=(batch_size, 1), minval=0.0, maxval=0.1),
                                      discriminator(fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

        grads = disc_tape.gradient(loss_disc, discriminator.trainable_weights)
        opt_disc.apply_gradients(
            zip(grads, discriminator.trainable_weights)
        )

        ### Train Generator min log(1 - D(G(z)) <-> max log(D(G(z))
        with tf.GradientTape() as gen_tape:
            fake = generator(random_latent_vectors)
            output = discriminator(fake)
            loss_gen = loss_fn(tf.ones(batch_size, 1), output)

        grads = gen_tape.gradient(loss_gen, generator.trainable_weights)
        opt_gen.apply_gradients(
            zip(grads, generator.trainable_weights)
        )
    # Save the model at the end of each epoch
    checkpoint_manager.save()
    print(f"Checkpoint saved for epoch {epoch}")
