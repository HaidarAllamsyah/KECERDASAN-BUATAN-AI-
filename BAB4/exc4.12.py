# Exercise 4.12 - webcam dengan DenseNet (juga bisa model lain)
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import vgg16, resnet50, mobilenet, inception_v3, densenet
import numpy as np

# buka komentar SATU baris (model, ukuran gambar, preprocess selalu berpasangan)
#model = vgg16.VGG16(weights='imagenet');               size = 224; preprocess = vgg16.preprocess_input
#model = resnet50.ResNet50(weights='imagenet');         size = 224; preprocess = resnet50.preprocess_input
#model = mobilenet.MobileNet(weights='imagenet');       size = 224; preprocess = mobilenet.preprocess_input
#model = inception_v3.InceptionV3(weights='imagenet');  size = 299; preprocess = inception_v3.preprocess_input
model = densenet.DenseNet121(weights='imagenet');       size = 224; preprocess = densenet.preprocess_input   # BARU
#model = densenet.DenseNet201(weights='imagenet');      size = 224; preprocess = densenet.preprocess_input   # BARU

camera = cv2.VideoCapture(0)
while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break
    frame = cv2.cvtColor(cv2.resize(cam_frame, (size, size)), cv2.COLOR_BGR2RGB)
    image_batch = np.expand_dims(img_to_array(frame), axis=0)
    predictions = model.predict(preprocess(image_batch.copy()), verbose=0)
    label = decode_predictions(predictions)
    cv2.putText(cam_frame, "{}, {:.1f}".format(label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('video image', cam_frame)
    if cv2.waitKey(30) == 27:       # ESC
        break

camera.release()
cv2.destroyAllWindows()
