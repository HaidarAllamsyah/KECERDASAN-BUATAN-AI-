# Exercise 4.15 - tampilkan filter layer konvolusi kedua (n = 2)
from keras.applications.vgg19 import VGG19
from matplotlib import pyplot

model = VGG19()
n = 2                                    # <-- diubah dari 1 menjadi 2 (block1_conv2)
filters, biases = model.layers[n].get_weights()
s = filters.shape                        # (3, 3, 64, 64) = (tinggi, lebar, channel_masuk, jumlah_filter)
print(model.layers[n].name, s)
print("Kernel:", s[0], "x", s[1], "| channel masuk:", s[2], "| jumlah filter:", s[3])

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters, n_channels = 4, 6             # 4 filter pertama, 6 channel pertama dari 64
ix = 1
pyplot.figure(figsize=(10, 6))
for i in range(n_filters):
    f = filters[:, :, :, i]              # satu filter: (3, 3, 64)
    for j in range(n_channels):
        ax = pyplot.subplot(n_filters, n_channels, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[:, :, j], cmap='gray')   # [FIX] channel ke-j, bukan f[j, :, :]
        ix += 1
pyplot.show()
