import cv2 as cv

model = cv.dnn.readNetFromTensorflow('Weights/my_model.h5')

img = cv.imread("Finale/Source/1538647057_1-2.jpg")

#cv.imshow("1", img)

# Use the given image as input, which needs to be blob(s).
img
blob = cv.dnn.blobFromImage(img, size=(720, 480), ddepth=cv.CV_32F, swapRB=True, crop=False)
model.setInput(blob)
# Runs a forward pass to compute the net output
out = model.forward()
print(1)
cv.imshow("3", out)

cv.waitKey(0)