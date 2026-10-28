# Exercise 4.16 - semua 64 filter layer 1 (n_filters = s[3])
from keras.applications.vgg19 import VGG19
from matplotlib import pyplot

model = VGG19()
n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape                                     # (3, 3, 3, 64)
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters = s[3]                                      # <-- semua filter (64)
col = 8
row = (n_filters + col - 1) // col                    # 8 baris
pyplot.figure(figsize=(10, 10))
for i in range(n_filters):
    ax = pyplot.subplot(row, col, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    pyplot.imshow(filters[:, :, :, i])                # tampilkan sebagai gambar berwarna RGB 3x3
pyplot.show()
