import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

# Membangun model GAN
def build_generator(latent_dim):
    model = keras.Sequential()
    model.add(layers.Dense(128, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(3072, activation='tanh'))
    model.add(layers.Reshape((32, 32, 3)))
    return model

# Membangun model discriminator
def build_discriminator(img_shape):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=img_shape))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Menggabungkan generator dan discriminator menjadi GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Melatih GAN dengan dataset gambar
def train_gan(generator, discriminator, gan, dataset, latent_dim, n_epochs=10000, batch_size=64):
    for epoch in range(n_epochs):
        for _ in range(dataset.shape[0] // batch_size):
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            generated_images = generator.predict(noise)
            real_images = dataset[np.random.randint(0, dataset.shape[0], batch_size)]

            # Menghitung loss discriminator
            d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Menghitung loss generator
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        print(f"Epoch: {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

        if epoch % 100 == 0:
            save_generated_image(epoch, generator)

# Simpan gambar yang dihasilkan oleh generator
def save_generated_image(epoch, generator, latent_dim=100):
    noise = np.random.normal(0, 1, (1, latent_dim))
    generated_image = generator.predict(noise)
    generated_image = 0.5 * generated_image + 0.5  # Rescale dari [-1, 1] ke [0, 1]
    generated_image = (generated_image * 255).astype(np.uint8)
    image = Image.fromarray(generated_image[0])
    image.save(f"generated_image_epoch_{epoch}.png")

if __name__ == '__main__':
    # Dataset gambar untuk melatih GAN (misalnya, gambar NFT yang ada)
    dataset = np.load('nft_dataset.npy')

    latent_dim = 100
    img_shape = (32, 32, 3)

    generator = build_generator(latent_dim)
    discriminator = build_discriminator(img_shape)
    gan = build_gan(generator, discriminator)

    train_gan(generator, discriminator, gan, dataset, latent_dim)
