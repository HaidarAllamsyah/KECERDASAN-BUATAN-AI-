# Exercise 4.13 - bandingkan ukuran & kecepatan beberapa model
import time
import numpy as np
from tensorflow.keras.applications import (VGG16, VGG19, ResNet50, MobileNetV2,
                                           InceptionV3, DenseNet201)

print(f"{'model':13s}{'input':>7s}{'parameter(juta)':>17s}{'layer':>7s}{'detik/gambar':>14s}")
for name, ctor in [("VGG16", VGG16), ("VGG19", VGG19), ("ResNet50", ResNet50),
                   ("MobileNetV2", MobileNetV2), ("InceptionV3", InceptionV3),
                   ("DenseNet201", DenseNet201)]:
    m = ctor(weights=None)                       # weights='imagenet' jika ingin akurasi nyata
    x = np.random.rand(1, *m.input_shape[1:]).astype('float32')
    m.predict(x, verbose=0)                      # pemanasan
    t0 = time.time()
    for _ in range(5):
        m.predict(x, verbose=0)
    dt = (time.time() - t0) / 5
    print(f"{name:13s}{m.input_shape[1]:>7d}{m.count_params()/1e6:>17.2f}{len(m.layers):>7d}{dt:>14.3f}")
