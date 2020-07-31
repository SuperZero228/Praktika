from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

#Тут нужно закинуть в список имена всех фотографий
SOURCE_PATH = "Finale/Source/"
MARKED_PATH = "Finale/Marked/"
#all_source_image_paths = os.listdir(SOURCE_PATH)
direct = os.getcwd()
all_source_image_paths = [SOURCE_PATH + str(path) for path in os.listdir(SOURCE_PATH)]
all_marked_image_paths = [MARKED_PATH + str(path) for path in os.listdir(MARKED_PATH)]
print(all_source_image_paths)
image_count = len(all_source_image_paths)

def preprocess_image(image):
  image = tf.image.decode_jpeg(image)
  image = tf.image.resize(image, [720, 480])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

def check(image_ds):
    import matplotlib.pyplot as plt

    for n, image in enumerate(image_ds.take(2)):
      fig, ax = plt.subplots()
      ax.imshow(image)
      fig.set_figwidth(6)  # ширина и
      fig.set_figheight(6)  # высота "Figure"
      plt.show()



path_source_ds = tf.data.Dataset.from_tensor_slices(all_source_image_paths)
source_image_ds = path_source_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

path_marked_ds = tf.data.Dataset.from_tensor_slices(all_marked_image_paths)
marked_image_ds = path_marked_ds.map(load_and_preprocess_image,  num_parallel_calls=AUTOTUNE)

#check(marked_image_ds)

final_ds = tf.data.Dataset.zip((source_image_ds, marked_image_ds))
print(final_ds)


# source_image_ds, marked_image_ds = final_ds
# check(marked_image_ds)





BATCH_SIZE = 1

# Установка размера буфера перемешивания, равного набору данных, гарантирует
# полное перемешивание данных.
ds = final_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
ds = ds.prefetch(buffer_size=AUTOTUNE)
print(ds)

model = tf.keras.Sequential([
    #encoder
    tf.keras.Input(shape=(720, 480, 3), name="Input"),
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1, activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),

    #decoder
    tf.keras.layers.Conv2D(filters=8,  kernel_size=3, strides=1, activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D(size=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D(size=2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, activation='sigmoid', padding='same')
])

model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


steps_per_epoch=tf.math.ceil(len(all_source_image_paths)/BATCH_SIZE).numpy()
print(steps_per_epoch)

model.fit(ds, epochs=1000, steps_per_epoch=2, batch_size=1)
model.evaluate(ds)