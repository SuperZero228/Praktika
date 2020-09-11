import cv2 as cv
import tensorflow as tf
import numpy as np

def preprocess_image(image):
  image = tf.image.decode_jpeg(image)
  image = tf.image.resize(image, [480, 720])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


model = tf.keras.models.load_model('Weights/model_h5/model.h5')
image = load_and_preprocess_image("Finale_images/Source/iWUNIEB47.jpg")
# print(image)
np = np.array([image])
out = model.predict(np)
out = out[0]
out *= 255


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.imshow(out)
fig.set_figwidth(6)  # ширина и
fig.set_figheight(6)  # высота "Figure"
plt.show()




# Надо как то конвертировать

model = cv.dnn.readNetFromTensorflow("Weights/test_model/simple_frozen_graph_v2.pb")
# model = cv.dnn.readNetFromTensorflow('Weights/model_h5/model.h5')
print(1)
img = cv.imread("Finale_images/Source/iWUNIEB47.jpg")
print(model)


# Нормализовать
# Use the given image as input, which needs to be blob(s).
print(img.shape)
blob = cv.dnn.blobFromImage(img, size=(720, 480), crop=False)
print(blob.shape)
model.setInput(blob)
# Runs a forward pass to compute the net output
out = model.forward()
print(out.shape)
out = out[0]
print(out.shape)
out = out[0]
print(out.shape)
print(out)
# for vector in out:
#   print(vector)
_, thresh1 = cv.threshold(out, 127, 255, cv.THRESH_BINARY)
cv.imshow("1", out)
cv.imshow("2", thresh1)
cv.waitKey(0)