# Exercise 4.14 - Image-Classifiers + pemilihan model dengan if-else
import cv2
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers

MODEL = 'resnet50'          # pilihan: 'vgg16', 'resnet50', 'mobilenetv2', 'densenet201', 'inceptionv3'

if MODEL == 'vgg16':
    clf, preprocess_input = Classifiers.get('vgg16');       sz = 224
    model = clf(input_shape=(sz, sz, 3), weights='imagenet', classes=1000)
elif MODEL == 'resnet50':
    clf, preprocess_input = Classifiers.get('resnet50');    sz = 224
    model = clf(input_shape=(sz, sz, 3), weights='imagenet', classes=1000)
elif MODEL == 'inceptionv3':
    clf, preprocess_input = Classifiers.get('inceptionv3'); sz = 299
    model = clf(input_shape=(sz, sz, 3), weights='imagenet', classes=1000)
elif MODEL == 'mobilenetv2':      # di Keras 3 library di atas gagal -> pakai keras.applications
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    sz = 224
    model = MobileNetV2(weights='imagenet')
elif MODEL == 'densenet201':      # idem
    from tensorflow.keras.applications import DenseNet201
    from tensorflow.keras.applications.densenet import preprocess_input
    sz = 224
    model = DenseNet201(weights='imagenet')
else:
    raise ValueError("MODEL tidak dikenal: " + MODEL)

print("Model dipakai:", MODEL, "| ukuran input:", sz)

camera = cv2.VideoCapture(0)
while True:
    ret, cam_frame = camera.read()
    if not ret:
        break
    frame = cv2.cvtColor(cv2.resize(cam_frame, (sz, sz)), cv2.COLOR_BGR2RGB)
    image = preprocess_input(np.expand_dims(np.asarray(frame), 0))
    label = decode_predictions(model.predict(image, verbose=0))
    cv2.putText(cam_frame, "{}: {}, {:.1f}".format(MODEL, label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", cam_frame)
    if cv2.waitKey(30) == 27:       # ESC
        break

camera.release()
cv2.destroyAllWindows()
