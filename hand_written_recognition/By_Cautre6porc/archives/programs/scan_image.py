""" Convert images into two array of data """

import json
from PIL import Image


def get_train():

    print('Scanning training data...')

    image = Image.open('train.png')
    pixels = image.load()
    train = []

    for digit in range(10):
        for row in range(73):
            for column in range(73):
                inp = []
                out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                out[digit] = 1
                for i in range(28):
                    for j in range(28):
                        gray = round(pixels[digit * 2044 + column * 28 + j, row * 28 + i] / 255, 1)
                        if gray == 0 or gray == 1:
                            gray = int(gray)
                        inp.append(gray)
                train.append((inp, out))

    mf = open('train.txt', 'w')
    mf.write(json.dumps(train))


def get_test():

    print('Scanning testing data...')

    image = Image.open('test.png')
    pixels = image.load()
    test = []

    for digit in range(10):
        for row in range(29):
            for column in range(29):
                inp = []
                out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                out[digit] = 1
                for i in range(28):
                    for j in range(28):
                        gray = round(pixels[digit * 812 + column * 28 + j, row * 28 + i] / 255, 1)
                        if gray == 0 or gray == 1:
                            gray = int(gray)
                        inp.append(gray)
                test.append((inp, out))

    mf = open('test.txt', 'w')
    mf.write(json.dumps(test))
