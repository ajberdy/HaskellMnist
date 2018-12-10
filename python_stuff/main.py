import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


from python_stuff.load_mnist import load_mnist
import python_stuff.FeatureExtraction as fe


def main(argv: []):
    train_images, train_labels, test_images, test_labels = load_mnist()
    image = train_images[1]
    pgl = fe.PixelgramLearner(epsilon=.9)   # very generous epsilon
    pgl.learn_zero_order(image)
    fe.show_filters(image, pgl.known_grams, pgl._weights)


if __name__ == '__main__':
    main(sys.argv)
