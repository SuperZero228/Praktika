from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os


AUTOTUNE = tf.data.experimental.AUTOTUNE


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):

    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    if print_graph == True:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
            print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def save_frozen_graph(model2save, path2save, version=2, as_text=False):

    # Create frozen graph
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    # Save model to SavedModel format
    path = path2save + "/model"
    tf.saved_model.save(model2save, path)

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model2save(x))
    full_model = full_model.get_concrete_function(x=tf.TensorSpec(model2save.inputs[0].shape, model2save.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=path2save,
                      name="simple_frozen_graph_v2.pb",
                      as_text=False)
    if as_text:
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=path2save,
                          name="simple_frozen_graph_v2.pbtxt",
                          as_text=True)

    if version == 1:
        # Load frozen graph using TensorFlow 1.x functions
        load_path = path2save + "/simple_frozen_graph_v2.pb"
        with tf.io.gfile.GFile(load_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            loaded = graph_def.ParseFromString(f.read())

        # Wrap frozen graph to ConcreteFunctions
        frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                        inputs=["x:0"],
                                        outputs=["Identity:0"],
                                        print_graph=True)

        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=path2save,
                          name="simple_frozen_graph_v1.pb",
                          as_text=True)
        if as_text:
            tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                              logdir=path2save,
                              name="simple_frozen_graph_v1.pbtxt",
                              as_text=True)


def preprocess_image(image):
  image = tf.image.decode_jpeg(image)
  image = tf.image.resize(image, [480, 720])
  image /= 255.0  # normalize to [0,1] range
  # image *= 2
  # image -= 1
  return image


def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


def create_dataset():

    # Тут нужно закинуть в список имена всех фотографий
    BATCH_SIZE = 10
    SOURCE_PATH = "Finale_images/Source/"
    MARKED_PATH = "Finale_images/Marked/"
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


def create_model():
    model = tf.keras.Sequential([
        # encoder
        tf.keras.Input(shape=(480, 720, 3), name="Input"),
        tf.keras.layers.Conv2D(filters=32, kernel_size=15, strides=1, activation='relu', padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),
        tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),

        # decoder
        tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D(size=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D(size=2),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D(size=2),
        tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, activation='relu', padding='same'),
    ])
    model.compile(optimizer='Adam',
                  loss=tf.keras.losses.mean_squared_logarithmic_error,
                  metrics=["BinaryAccuracy"])

    return model


ds = create_dataset()

model = create_model()

model.fit(ds, epochs=200, steps_per_epoch=5)
model.evaluate(ds, steps=10)


# Сохраним всю модель в  HDF5 файл
model.save('Weights/model_h5/model.h5')

# Сохраняем веса
model.save_weights('Weights/checkpoints/my_checkpoint_weights')

# Сохраняем замороженный граф
save_frozen_graph(model, path2save="Weights/test_model", version=2, as_text=True)
