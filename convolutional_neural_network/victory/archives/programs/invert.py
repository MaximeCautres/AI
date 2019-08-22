import matplotlib.pyplot as plt
import numpy as np
import pickle

data = pickle.load(open('cifar_15_sorted', 'rb'))

train_x_ = data['train_x']
train_y_ = data['train_y']
test_x_ = data['test_x']
test_y_ = data['test_y']

train_x = np.zeros((32, 32, 3, 15000), dtype=np.float16)
train_y = np.zeros((15, 15000), dtype=np.bool)
train_x[:, :, :, :7500] = train_x_
train_y[:, :7500] = train_y_

for k in range(7500):
    train_x[:, :, :, 7500+k] = np.flip(train_x_[:, :, :, k], 1)
    train_y[:, 7500+k] = train_y_[:, k]

test_x = np.zeros((32, 32, 3, 3000), dtype=np.float16)
test_y = np.zeros((15, 3000), dtype=np.bool)
test_x[:, :, :, :1500] = test_x_
test_y[:, :1500] = test_y_

for k in range(1500):
    test_x[:, :, :, 1500+k] = np.flip(test_x_[:, :, :, k], 1)
    test_y[:, 1500+k] = test_y_[:, k]

data = {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y}

pickle.dump(data, open('cifar_15_sorted_inverted', 'wb'))

def show(index):
    img = np.array(train_x[:, :, :, index], dtype=np.float32)

    print(train_y[:, index])

    plt.imshow(img)