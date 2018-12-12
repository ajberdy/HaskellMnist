#!/usr/bin/env python

import sys, os

from sklearn import svm, metrics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.load_mnist import load_mnist
import src.FeatureExtraction as fe


def main(argv: []):
    train_images, train_labels, test_images, test_labels = load_mnist()
    # image = train_images[1]
    # pgl = fe.PixelgramLearner(epsilon=.9)   # very generous epsilon
    # pgl.learn_zero_order([image])
    # fe.show_filters(image, pgl.known_grams, pgl._weights)

    pgl2 = fe.PixelgramLearner(epsilon=.1)
    pgl2.learn_zero_order_pos(train_images[:10])

    # fe.show_pos_filters(image, pgl2.known_grams, pgl2._weights)
    # fe.visualize_pos_alignments()
    # input(':')
    for img in train_images[10:20]:
        fe.show_pos_filters(img, pgl2.known_grams, pgl2._weights)


if __name__ == '__main__':
    main(sys.argv)
