import pickle
import numpy as np
import matplotlib.pyplot as plt


data = pickle.load(open('cifar_10_int', 'rb'))

train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']

# labels = ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish',
#           'ray', 'shark', 'trout', 'sea', 'crab', 'lobster', 'crocodile', 'ship']
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

data = {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y, 'labels': labels}
# pickle.dump(data, open('mnist_int', 'wb'))


def show(index):
    pixels = train_x[:, :, :, index]

    print(train_y[:, index])

    plt.close()
    plt.imshow(pixels)
    plt.show()
