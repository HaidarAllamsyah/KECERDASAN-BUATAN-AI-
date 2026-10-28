# Exercise 4.17 - feature map untuk layer konvolusi ke-2, ke-3, ke-4
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.utils import load_img, img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims

def plot_feature_maps(feature_maps, max_maps=64):     # batasi 64 peta supaya figure terbaca
    col = 8
    total = min(feature_maps.shape[3], max_maps)
    row = total // col
    ix = 1
    plt.figure(figsize=(16, 16))
    for _ in range(row):
        for _ in range(col):
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    plt.show()

base = VGG19()
# indeks layer konvolusi: 1, 2, 4, 5, 7, ...  (indeks 3 adalah pooling, bukan konvolusi)
img = load_img('Elephant.jpg', target_size=(224, 224))
img = preprocess_input(expand_dims(img_to_array(img), axis=0))

for n in (2, 4, 5):                                   # konvolusi ke-2, ke-3, ke-4
    sub = Model(inputs=base.inputs, outputs=base.layers[n].output)
    feature_maps = sub.predict(img)
    print("layer", n, base.layers[n].name, "feature maps:", feature_maps.shape)
    plot_feature_maps(feature_maps)
