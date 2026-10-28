#Example 4.12
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import numpy as np

from tensorflow.keras.applications import (
        vgg16,
        resnet50,
        mobilenet,
        inception_v3
    )

# init the models  -- buka komentar SATU model saja, dan samakan 3 baris: model, size, preprocess
#model = vgg16.VGG16(weights='imagenet');               size = 224; preprocess = vgg16.preprocess_input
#model = resnet50.ResNet50(weights='imagenet');         size = 224; preprocess = resnet50.preprocess_input
model = mobilenet.MobileNet(weights='imagenet');        size = 224; preprocess = mobilenet.preprocess_input
#model = inception_v3.InceptionV3(weights='imagenet');  size = 299; preprocess = inception_v3.preprocess_input   # [FIX] size 299
print(model.summary())

img_path = 'Elephant.jpg'
img = image.load_img(img_path, target_size=(size, size))                  # [FIX] pakai size
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
processed_image = preprocess(x)

# prediction
predictions = model.predict(processed_image)                              # [FIX] buku memakai x
results = decode_predictions(predictions)
print(results)
