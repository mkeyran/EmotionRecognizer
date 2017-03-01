#!/usr/bin/env python

import cv2
import sys
import numpy as np


def colors_percent(image):
    rec = np.core.records.fromarrays(image.transpose(),
                            names='r, g, b', formats='i8, i8, i8')
    uniques = np.unique(rec.flatten())
    return (len(uniques)/len(rec.flatten()))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} <image file>".format(sys.argv[0]))
        sys.exit(1)
    image = cv2.imread(sys.argv[1])  # Читаем изображение
    if image is None:
        print("File not found")
        sys.exit(2)
    print(colors_percent(image))
