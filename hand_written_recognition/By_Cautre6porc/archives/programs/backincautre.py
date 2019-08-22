""" Transform a matrix into a image and the inverse """

import math
from PIL import Image


def save(matrix):

    data = matrix_to_array(matrix)

    v_min = 0
    v_max = 0
    for x in data:
        if isinstance(x, float):
            if v_min > x:
                v_min = x
            if v_max < x:
                v_max = x

    offset = (v_max - v_min) / 15000000
    v_min -= offset
    v_max += offset
    colors = []
    cc = 255 * 255
    c = 255

    for x in data:
        if isinstance(x, float) or isinstance(x, int):
            x = (x - v_min) * 16581375 / (v_max - v_min)
            r = 0
            while x >= cc:
                r += 1
                x -= cc
            g = 0
            while x >= c:
                g += 1
                x -= c
            b = int(x)
            color = (r, g, b)
        elif x == '[':
            color = (255, 255, 255)
        else:
            color = (0, 0, 0)

        colors.append(color)

    r = int(math.ceil(math.sqrt(len(colors))))

    image = Image.new('RGB', (r, r))
    pixels = image.load()

    for i in range(len(colors)):
        x = i % r
        y = i // r
        pixels[x, y] = colors[i]

    name = str(v_min) + '_' + str(v_max) + '.png'
    image.save(name)


def load(image, name):

    name = name[:-4]
    v_min, v_max = name.split('_')
    v_min = float(v_min)
    v_max = float(v_max)

    pixels = image.load()
    r = image.size[0]
    colors = []

    for x in range(r):
        for y in range(r):
            colors.append(pixels[y, x])

    data = []
    count = 0

    for color in colors:
        if color == (255, 255, 255):
            data.append('[')
            count += 1
        elif color == (0, 0, 0):
            data.append(']')
            count -= 1
        else:
            r, g, b = color
            x = r * 255 * 255 + g * 255 + b
            data.append(x * (v_max - v_min) / 16581375 + v_min)

        if count == -1:
            del data[-1:]
            break

    matrix = array_to_matrix(data)

    return matrix


def matrix_to_array(matrix):

    stack = []

    for x in matrix:
        if isinstance(x, list):
            stack += ['['] + x + [']']
        else:
            stack.append(x)

    for x in stack:
        if isinstance(x, list):
            stack = matrix_to_array(stack)
            break

    return stack


def array_to_matrix(array):

    matrix = []
    stack = []
    begin = False

    for i in range(len(array)):

        x = array[i]
        matrix.append(x)

        if x == '[':
            begin = True
            stack = []
        elif x == ']':
            if begin:
                del matrix[- len(stack) - 2:]
                matrix.append(stack)
                begin = False
        else:
            stack.append(x)

    for x in matrix:
        if isinstance(x, str):
            matrix = array_to_matrix(matrix)
            break

    return matrix
