import matplotlib.pyplot as plt
import numpy as np
import pickle

data = pickle.load(open('cifar_100_coarse', 'rb'))

train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']
labels = data['labels']

sorted_train_x = np.zeros((32, 32, 3, 2500, 20), dtype=np.uint8)
sorted_train_y = np.zeros((20, 2500, 20), dtype=np.bool)

counter = np.zeros((20, ))
for i in range(50000):
    for k in range(20):
        if train_y[k, i]:
            count = int(counter[k])
            sorted_train_x[:, :, :, count, k] = train_x[:, :, :, i]
            sorted_train_y[:, count, k] = train_y[:, i]
            counter[k] = count + 1

train_x = sorted_train_x.reshape((32, 32, 3, 50000))
train_y = sorted_train_y.reshape((20, 50000))

sorted_test_x = np.zeros((32, 32, 3, 500, 20), dtype=np.uint8)
sorted_test_y = np.zeros((20, 500, 20), dtype=np.bool)

counter = np.zeros((20, ))
for i in range(10000):
    for k in range(20):
        if test_y[k, i]:
            count = int(counter[k])
            sorted_test_x[:, :, :, count, k] = test_x[:, :, :, i]
            sorted_test_y[:, count, k] = test_y[:, i]
            counter[k] = count + 1

test_x = sorted_test_x.reshape((32, 32, 3, 10000))
test_y = sorted_test_y.reshape((20, 10000))

data = {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y, 'labels': labels}
pickle.dump(data, open('cifar_100_sorted', 'wb'))

def show(index):
    pixels = train_x[:, :, :, index]

    print(train_y[:, index])

    plt.close()
    plt.imshow(np.array(pixels))
    plt.show()
