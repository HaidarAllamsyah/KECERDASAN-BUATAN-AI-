#Example 4.10
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions   # [FIX] versi tf.keras
import numpy as np

model = VGG16(weights='imagenet')
print(model.summary())
