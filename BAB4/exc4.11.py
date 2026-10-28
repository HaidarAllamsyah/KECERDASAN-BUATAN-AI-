# Exercise 4.11 - webcam + VGG19
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import vgg19
import numpy as np

image_size = 224
model = vgg19.VGG19(weights='imagenet')

camera = cv2.VideoCapture(0)
while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break
    frame = cv2.resize(cam_frame, (image_size, image_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_batch = np.expand_dims(img_to_array(frame), axis=0)
    processed_image = vgg19.preprocess_input(image_batch.copy())

    predictions = model.predict(processed_image, verbose=0)
    label = decode_predictions(predictions)

    cv2.putText(cam_frame, "VGG19: {}, {:.1f}".format(label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('video image', cam_frame)
    if cv2.waitKey(30) == 27:       # ESC
        break

camera.release()
cv2.destroyAllWindows()
