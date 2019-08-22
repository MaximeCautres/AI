""" Show digit which is in the data set """

import json
import random
import numpy as np
import matplotlib.pyplot as plt

print('Loading...')

" Load data "

data_file = open('data.txt', 'r')
data = json.loads(data_file.read())
data_file.close()

" Condition data "

train_x = np.array(data['train_x'])
train_y = np.array(data['train_y'])

test_x = np.array(data['test_x'])
test_y = np.array(data['test_y'])

print('Ready !')

for i in range(10):
    image = np.zeros((28, 28, 3), dtype=np.uint8)

    index = random.randint(0, 50000)
    pixels = train_x[:, index]

    print(train_y[:, index])

    for j in range(28):
        for k in range(28):
            pixel = pixels[j * 28 + k]
            gray = int(pixel * 255)
            image[j][k] = (gray, gray, gray)

    plt.imshow(image)
    plt.show()
