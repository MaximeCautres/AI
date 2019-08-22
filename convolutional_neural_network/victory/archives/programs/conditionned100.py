import pickle
import numpy as np
import matplotlib.pyplot as plt


meta = pickle.load(open('meta', 'rb'), encoding='bytes')
meta_y = meta[b'fine_label_names']
meta_z = meta[b'coarse_label_names']

num = 20

train = pickle.load(open('train', 'rb'), encoding='bytes')
train_x_ = train[b'data'].reshape((50000, 3, 32, 32)).T
train_y_ = np.array(train[b'fine_labels'])
train_z_ = np.array(train[b'coarse_labels'])

count = 0
train_x = np.zeros((32, 32, 3, 50000), dtype=np.uint8)
train_z = np.zeros((num, 50000), dtype=np.bool)
for i in range(50000):
    k = train_z_[i]
    label = np.zeros((num,), dtype=np.bool)
    label[k] = True
    train_x[:, :, :, count] = np.swapaxes(train_x_[:, :, :, i], 0, 1)
    train_z[:, count] = label
    count += 1

test = pickle.load(open('test', 'rb'), encoding='bytes')
test_x_ = (test[b'data']).reshape((10000, 3, 32, 32)).T
test_y_ = np.array(test[b'fine_labels'])
test_z_ = np.array(test[b'coarse_labels'])

count = 0
test_x = np.zeros((32, 32, 3, 10000), dtype=np.uint8)
test_z = np.zeros((num, 10000), dtype=np.bool)
for i in range(10000):
    k = test_z_[i]
    label = np.zeros((num,), dtype=np.bool)
    label[k] = True
    test_x[:, :, :, count] = np.swapaxes(test_x_[:, :, :, i], 0, 1)
    test_z[:, count] = label
    count += 1

labels_y = [x.decode() for x in meta_y]
labels_z = [x.decode() for x in meta_z]

print(labels_z)

data = {'train_x': train_x, 'train_y': train_z, 'test_x': test_x, 'test_y': test_z, 'labels': labels_z}
pickle.dump(data, open('cifar_100_coarse', 'wb'))

def show(index):
    pixels = train_x[:, :, :, index]

    print(train_z[:, index])

    plt.close()
    plt.imshow(np.array(pixels))
    plt.show()
