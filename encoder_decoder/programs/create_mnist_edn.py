import matplotlib.pyplot as plt
import numpy as np
import pickle


def rt():
    return tuple(np.random.randint(28, size=256))


data = pickle.load(open('mnist', 'rb'))

train_y = data['train_x']
test_y = data['test_x']
del data['labels']

train_x = np.copy(train_y)
for k in range(train_x.shape[3]):
    train_x[rt(), rt(), :, k] = train_y[rt(), rt(), :, k]

test_x = np.copy(test_y)
for k in range(test_x.shape[3]):
    test_x[rt(), rt(), :, k] = test_y[rt(), rt(), :, k]

data = {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y}

pickle.dump(data, open('mnist_edn', 'wb'))


def show(index):
    img_x = train_x[:, :, 0, index]
    img_y = train_y[:, :, 0, index]

    plt.close()
    plt.imshow(np.concatenate((img_x, img_y), axis=1))
