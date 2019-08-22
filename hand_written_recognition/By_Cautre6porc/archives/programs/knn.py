""" The KNN method """

import scan_image as si

data = si.get_test()


def process(array):

    minimum = 10 ** 6
    prediction = 0

    for num in range(10):
        for digit in range(841):
            errors = []
            for i in range(784):
                errors.append((array[i] - data[num * 841 + digit][0][i]) ** 2)
            if sum(errors) < minimum:
                minimum = sum(errors)
                prediction = num

    return prediction
