import cv2 as cv
import tensorflow as tf
import numpy as np

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


print(tf.train.list_variables('Weights/check/my_checkpoint_weights'))



model = tf.keras.models.load_model('Weights/my_model.h5')
print(model.get_weights())
image = load_and_preprocess_image("Finale/Source/1.jpg")
# print(image)
np = np.array([image])
print(np)
out = model.predict(np)
out = out[0]
out += 1
out /= 2
out *= 255
print(out)
cv.imshow("1", out)
cv.waitKey()

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.imshow(out)
fig.set_figwidth(6)  # ширина и
fig.set_figheight(6)  # высота "Figure"
plt.show()




# # Надо как то конвертировать
# cv.dnn.writeTextGraph('Weights/my_model/saved_model.pb', 'Weights/')
#
# model = cv.dnn.readNetFromTensorflow('Weights/my_model.h5')
#
# img = cv.imread("Finale/Source/1538647057_1-2.jpg")
#
# #cv.imshow("1", img)
#
# # Use the given image as input, which needs to be blob(s).
# img
# blob = cv.dnn.blobFromImage(img, size=(720, 480), ddepth=cv.CV_32F, swapRB=True, crop=False)
# model.setInput(blob)
# # Runs a forward pass to compute the net output
# out = model.forward()
# print(1)
# cv.imshow("3", out)
#
# cv.waitKey(0)