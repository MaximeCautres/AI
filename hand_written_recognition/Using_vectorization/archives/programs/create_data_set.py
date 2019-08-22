""" Scan and prepare data """

import json
from PIL import Image


def transpose(matrix, r, c):
    new = []

    for y in range(c):
        new.append([])
        for x in range(r):
            new[y].append(matrix[x][y])

    return new


all_data = [[] for _ in range(10)]

" Train part "

train_img = Image.open('train.png')
train_px = train_img.load()

for digit in range(10):
        for row in range(73):
            for column in range(73):
                inp = []
                out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                out[digit] = 1
                for i in range(28):
                    for j in range(28):
                        gray = round(train_px[digit * 2044 + column * 28 + j, row * 28 + i] / 255, 1)
                        if gray == 0 or gray == 1:
                            gray = int(gray)
                        inp.append(gray)
                all_data[digit].append({'input': inp, 'output': out})

" Test part "

test_img = Image.open('test.png')
test_px = test_img.load()

for digit in range(10):
        for row in range(29):
            for column in range(29):
                inp = []
                out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                out[digit] = 1
                for i in range(28):
                    for j in range(28):
                        gray = round(test_px[digit * 812 + column * 28 + j, row * 28 + i] / 255, 1)
                        if gray == 0 or gray == 1:
                            gray = int(gray)
                        inp.append(gray)
                all_data[digit].append({'input': inp, 'output': out})

train_x = []
train_y = []

test_x = []
test_y = []

for i in range(0, 5000):
    for d in range(10):
        train_x.append(all_data[d][i]['input'])
        train_y.append(all_data[d][i]['output'])

for i in range(5000, 6000):
    for d in range(10):
        test_x.append(all_data[d][i]['input'])
        test_y.append(all_data[d][i]['output'])


train_x = transpose(train_x, 50000, 784)
train_y = transpose(train_y, 50000, 10)

test_x = transpose(test_x, 10000, 784)
test_y = transpose(test_y, 10000, 10)

data = {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y}

my_file = open('data.txt', 'w')
my_file.write(json.dumps(data))
my_file.close()