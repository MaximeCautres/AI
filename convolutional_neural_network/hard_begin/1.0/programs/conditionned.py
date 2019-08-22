import pickle
import numpy as np
import matplotlib.pyplot as plt

data = pickle.load(open('old.p', 'rb'))
train_x, train_y = data['train_x'], data['train_y']
test_x, test_y = data['test_x'], data['test_y']

train_x = train_x.reshape(28, 28, 1, 50000)
test_x = test_x.reshape(28, 28, 1, 10000)

data = {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y}
pickle.dump(data, open('data.p', 'wb'))


def show(index):
    image = np.zeros((28, 28, 3))
    pixels = train_x[:, :, :, index]

    print(train_y[:, index])

    for w in range(28):
        for h in range(28):
            image[w][h] = (pixels[w, h, 0])

    plt.close()
    plt.imshow(image)
    plt.show()
