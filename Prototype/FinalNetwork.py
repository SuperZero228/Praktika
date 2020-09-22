from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os


class DataSetCreator:

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def __pre_process_image(self, image):
        image = tf.image.decode_jpeg(image)
        image = tf.image.resize(image, [480, 720])
        image /= 255.0  # normalize to [0,1] range
        return image

    def load_and_pre_process_image(self, path):
        image = tf.io.read_file(path)
        return self.__pre_process_image(image)

    def create_data_set(self, source, marked, batch_size):
        # Тут нужно закинуть в список имена всех фотографий
        # BATCH_SIZE = 10
        # SOURCE_PATH = "Finale_images/Source/"
        # MARKED_PATH = "Finale_images/Marked/"
        direct = os.getcwd()
        all_source_image_paths = [SOURCE_PATH + str(path) for path in os.listdir(SOURCE_PATH)]
        all_marked_image_paths = [MARKED_PATH + str(path) for path in os.listdir(MARKED_PATH)]
        image_count = len(all_source_image_paths)

        path_source_ds = tf.data.Dataset.from_tensor_slices(all_source_image_paths)
        source_image_ds = path_source_ds.map(self.load_and_pre_process_image, num_parallel_calls=self.AUTOTUNE)

        path_marked_ds = tf.data.Dataset.from_tensor_slices(all_marked_image_paths)
        marked_image_ds = path_marked_ds.map(self.load_and_pre_process_image, num_parallel_calls=self.AUTOTUNE)

        ds = tf.data.Dataset.zip((source_image_ds, marked_image_ds))

        # Установка размера буфера перемешивания, равного набору данных, гарантирует
        # полное перемешивание данных.
        ds = ds.shuffle(buffer_size=image_count)
        ds = ds.repeat()
        ds = ds.batch(BATCH_SIZE)
        # `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)

        return ds

#+++++++++++++++++++++++ВАЖНО++++++++++++++++++++++++++++
# Добавить 1 версию и усовершенствовать создание папкок, сделать рефакторинг папок
class Network:

    def __init__(self, name_of_model):
        self.model_name = name_of_model
        self.model = tf.keras.Sequential([
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
        self.model.compile(optimizer='Adam',
                      loss=tf.keras.losses.mean_squared_logarithmic_error,
                      metrics=["BinaryAccuracy"])

    def __wrap_frozen_graph(self, graph_def, inputs, outputs, print_graph=False):

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

    def save_frozen_graph(self, path2save, version=2, as_text=False):
        # Create frozen graph
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        # Save model to SavedModel format
        path = path2save + "/model"
        tf.saved_model.save(self.model, path)

        # Convert Keras model to ConcreteFunction
        full_model = tf.function(lambda x: self.model(x))
        full_model = full_model.get_concrete_function(x=tf.TensorSpec(self.model.inputs[0].shape, self.model.inputs[0].dtype))

        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        layers = [op.name for op in frozen_func.graph.get_operations()]

        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=path2save,
                          name=self.model_name + "_frozen_graph_v2.pb",
                          as_text=False)
        if as_text:
            tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                              logdir=path2save,
                              name=self.model_name + "_frozen_graph_v2.pbtxt",
                              as_text=True)

        if version == 1:
            # Load frozen graph using TensorFlow 1.x functions
            load_path = path2save + "/" + self.model_name + "_frozen_graph_v2.pb"
            with tf.io.gfile.GFile(load_path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                loaded = graph_def.ParseFromString(f.read())

            # Wrap frozen graph to ConcreteFunctions
            frozen_func = self.__wrap_frozen_graph(graph_def=graph_def,
                                            inputs=["x:0"],
                                            outputs=["Identity:0"],
                                            print_graph=True)

            tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                              logdir=path2save,
                              name=self.model_name + "_frozen_graph_v1.pb",
                              as_text=True)
            if as_text:
                tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                                  logdir=path2save,
                                  name=self.model_name + "_frozen_graph_v1.pbtxt",
                                  as_text=True)

    def save_weights(self, path2save):
        # Сохраняем веса
        path = path2save + "/" + self.model_name + "_weights"
        self.model.save_weights(path)

    def save_hdf5_format(self, path2save):
        # Сохраним всю модель в  HDF5 файл
        path = path2save + "/" + self.model_name + ".h5"
        self.model.save(path)

    def start_training(self, data_set, number_of_epochs, steps_per_epoch):
        self.model.fit(data_set, epochs=number_of_epochs, steps_per_epoch=steps_per_epoch)
        self.model.evaluate(data_set, steps=steps_per_epoch)




BATCH_SIZE = 10
SOURCE_PATH = "Finale_images/Source/"
MARKED_PATH = "Finale_images/Marked/"

creator = DataSetCreator()
ds = creator.create_data_set(source=SOURCE_PATH, marked=MARKED_PATH, batch_size=5)

net = Network(name_of_model="TEST")
net.start_training(data_set=ds, number_of_epochs=300, steps_per_epoch=10)
net.save_hdf5_format(path2save='Weights/models_h5/')
net.save_frozen_graph(path2save='Weights/frozen_models')
net.save_weights(path2save='Weights/weights')
