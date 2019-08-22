import pickle
import numpy as np
import matplotlib.pyplot as plt


data = pickle.load(open('cifar_10_int', 'rb'))

train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']
labels = data['label']

data = {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y}
# pickle.dump(data, open('data', 'wb'))


def show(index):
    pixels = train_x[:, :, :, index]

    print(train_y[:, index])

    plt.close()
    plt.imshow(np.array(pixels))
    plt.show()
