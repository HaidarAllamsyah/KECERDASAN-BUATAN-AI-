# Exercise 4.10 - bandingkan VGG16 vs VGG19 pada gambar yang sama
import time
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16, vgg19
from tensorflow.keras.applications.imagenet_utils import decode_predictions

img = image.load_img('Elephant.jpg', target_size=(224, 224))
x0 = np.expand_dims(image.img_to_array(img), axis=0)

for name, module, ctor in [("VGG16", vgg16, vgg16.VGG16), ("VGG19", vgg19, vgg19.VGG19)]:
    model = ctor(weights='imagenet')
    x = module.preprocess_input(x0.copy())
    model.predict(x, verbose=0)                       # pemanasan
    t0 = time.time()
    preds = model.predict(x, verbose=0)
    dt = time.time() - t0
    print(f"\n{name}: {model.count_params():,} parameter, {dt:.3f} detik/gambar")
    for _, label, p in decode_predictions(preds, top=5)[0]:
        print(f"   {label:28s}{p:.4f}")
