import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 as cv

def preprocess_image(image):
  image = tf.image.decode_jpeg(image)
  image = tf.image.resize(image, [480, 720])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


model = tf.keras.models.load_model('Weights/models_h5/TEST.h5')
image = load_and_preprocess_image("Finale_images/Source/iWUNIEB47.jpg")
# print(image)
np = np.array([image])
out = model.predict(np)
out = out[0]
out *= 255
fig, ax = plt.subplots()
ax.imshow(out)
fig.set_figwidth(6)
fig.set_figheight(6)
plt.show()



model = cv.dnn.readNetFromTensorflow("Weights/frozen_models/TEST_frozen_graph_v2.pb")
img = cv.imread("Finale_images/Source/iWUNIEB47.jpg")


# Use the given image as input, which needs to be blob(s).
blob = cv.dnn.blobFromImage(img, size=(720, 480), crop=False)
model.setInput(blob)

# Runs a forward pass to compute the net output
out = model.forward()

# Pull photo
out = out[0][0]

# for vector in out:
#   print(vector)
_, thresh = cv.threshold(out, 127, 255, cv.THRESH_BINARY)
cv.imshow("1", out)
cv.imshow("2", thresh)
cv.waitKey(0)