from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os


AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocess_image(image):
  image = tf.image.decode_jpeg(image)
  image = tf.image.resize(image, [720, 480])
  image /= 255.0  # normalize to [0,1] range
  image *= 2
  image -= 1
  return image


def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


def create_dataset():

    # Тут нужно закинуть в список имена всех фотографий
    BATCH_SIZE = 10
    SOURCE_PATH = "Finale/Source/"
    MARKED_PATH = "Finale/Marked/"
    # all_source_image_paths = os.listdir(SOURCE_PATH)
    direct = os.getcwd()
    all_source_image_paths = [SOURCE_PATH + str(path) for path in os.listdir(SOURCE_PATH)]
    all_marked_image_paths = [MARKED_PATH + str(path) for path in os.listdir(MARKED_PATH)]
    print(all_source_image_paths)
    image_count = len(all_source_image_paths)

    path_source_ds = tf.data.Dataset.from_tensor_slices(all_source_image_paths)
    source_image_ds = path_source_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    print(source_image_ds)

    path_marked_ds = tf.data.Dataset.from_tensor_slices(all_marked_image_paths)
    marked_image_ds = path_marked_ds.map(load_and_preprocess_image,  num_parallel_calls=AUTOTUNE)
    print(marked_image_ds)
    print(len(list(marked_image_ds.as_numpy_iterator())))


    ds = tf.data.Dataset.zip((source_image_ds, marked_image_ds))
    print(ds)


    # Установка размера буфера перемешивания, равного набору данных, гарантирует
    # полное перемешивание данных.
    ds = ds.shuffle(buffer_size=image_count)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    print(ds)

    return ds


ds = create_dataset()

CHECKPOINT_PATH = "Weights/cp.ckpt"


model = tf.keras.Sequential([
    #encoder
    tf.keras.Input(shape=(720, 480, 3), name="Input"),
    tf.keras.layers.Conv2D(filters=32, kernel_size=15, strides=1, activation='relu', padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),

    #decoder
    tf.keras.layers.Conv2D(filters=16,  kernel_size=3, strides=1, activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D(size=2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D(size=2),
    tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, activation='relu', padding='same'),
])


loss = tf.keras.losses
model.compile(optimizer='Adam',
              loss=loss.mean_squared_logarithmic_error,
              metrics=['accuracy'])


# Распечатаем архитектуру модели
model.summary()

print(model.activity_regularizer)


model.fit(ds, epochs=4000, steps_per_epoch=5)
model.evaluate(ds, steps=10)


# Сохраним всю модель в  HDF5 файл
model.save('Weights/my_model.h5')
model.save('Weights/my_model')
# Сохраняем веса
model.save_weights('Weights/check/my_checkpoint_weights')

