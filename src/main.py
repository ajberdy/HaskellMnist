#!/usr/bin/env python

import sys, os

from sklearn import svm, metrics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.load_mnist import load_mnist
import src.FeatureExtraction as fe

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier


def main(argv: []):
    train_images, train_labels, test_images, test_labels = load_mnist()
    image = train_images[1]
    # pgl = fe.PixelgramLearner(epsilon=.9)   # very generous epsilon
    # pgl.learn_zero_order([image])
    # fe.show_filters(image, pgl.known_grams, pgl._weights)

    pgl2 = fe.PixelgramLearner(epsilon=.1)
    pgl2.learn_zero_order_pos(train_images[:10])

    # X_train = np.loadtxt("src/training.csv", delimiter=',')
    # X_test = np.loadtxt("src/testing.csv", delimiter=',')
    #
    # y_train = np.argmax(train_labels[:100], axis=1)
    # y_test = np.argmax(test_labels[:100], axis=1)
    #
    # classifier = KNeighborsClassifier(n_neighbors=5)
    # classifier.fit(X_train, y_train)
    #
    # predicted = classifier.predict(X_test)
    #
    # print("Classification report for classifier %s:\n%s\n"
    #       % (classifier, metrics.classification_report(y_test, predicted)))
    # print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))
    #
    # images_and_predictions = list(zip(test_images[:100].reshape((100, 28, 28)), predicted))
    # for index, (image, prediction) in enumerate(images_and_predictions[:16]):
    #     plt.subplot(4, 4, 1 + index)
    #     plt.axis('off')
    #     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    #     plt.title('Prediction: %i' % prediction)
    # plt.show()

    for img in test_images[:10]:
        fe.show_basis_segmentation(img, pgl2)
    # fe.show_pos_filters(image, pgl2)
    # fe.visualize_pos_alignments()
    # input(':')
    # for img in train_images[10:20]:
    #     fe.show_pos_filters(img, pgl2.known_grams, pgl2._weights)

    # B = pgl2.known_grams
    # X_r = [image.flatten() for image in train_images[:100]]
    # print(X_r)
    # X_train = np.array([[np.sum([b.alignment(fe.PosPixel(p, ix)) for ix, p in enumerate(image)])
    #                for b in B] for image in X_r], dtype=float)
    # print(X_train)
    # df = pd.DataFrame(X_train)
    # df.to_csv("training.csv", header=None, index=None)
    #
    # X_train = np.loadtxt("training.csv", delimiter=',')
    # y_train = np.argmax(train_labels[:100], axis=1)
    # # print(y_train)
    # # print(train_labels[:10])
    #
    # classifier = svm.LinearSVC()
    # # print(X_train.shape, y_train.shape)
    # classifier.fit(X_train, y_train)
    #
    # expected = np.argmax(test_labels[:100], axis=1)
    # transformed = np.array([[np.sum([b.alignment(fe.PosPixel(p, ix)) for ix, p in enumerate(image.flatten())])
    #                for b in B] for image in test_images[:100]], dtype=float)
    # df = pd.DataFrame(transformed)
    # df.to_csv("testing.csv", header=None, index=None)

    # predicted = classifier.predict(transformed)
    #
    # print("Classification report for classifier %s:\n%s\n"
    #       % (classifier, metrics.classification_report(expected, predicted)))
    # print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    #
    # images_and_predictions = list(zip(test_images[:100].reshape((100, 28, 28)), predicted))
    # for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    #     plt.subplot(2, 4, index + 5)
    #     plt.axis('off')
    #     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    #     plt.title('Prediction: %i' % prediction)
    #     plt.show()


if __name__ == '__main__':
    main(sys.argv)
