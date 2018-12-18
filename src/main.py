#!/usr/bin/env python
"""
concept basis research pipeline
"""
import argparse
import logging
import signal
import sys

# if you get ModuleNotFoundError, run:
# $ export PYTHONPATH=$PYTHONPATH:`path/to/intelligence-basis-mnist`

from src.load_mnist import load_mnist
import src.FeatureExtraction as fe


LOGGER = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s")


def signal_handler(signum, frame):
    """ exit with ^C """
    del signum, frame
    print()
    LOGGER.info("Keyboard interrupt: exiting")
    sys.exit(0)


def parse_args(argv: []):
    """ parse args """
    parser = argparse.ArgumentParser(usage=__doc__.rstrip())
    parser.add_argument("--log-level", "-V", dest="log_level", action="store",
                        default="INFO", help="set the logging level")
    parser.add_argument("--epsilon", "-e", dest="epsilon", action="store",
                        default=.1, help="sensitivity parameter")
    args = parser.parse_args(argv)
    LOGGER.setLevel(args.log_level)
    LOGGER.debug("log-level set to %s", args.log_level)
    return args


def main(argv: []):
    """ main """
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args(argv[1:])
    epsilon = args.epsilon
    train_images, train_labels, test_images, test_labels = load_mnist()
    pgl2 = fe.PixelgramLearner(epsilon=epsilon)
    pgl2.learn_zero_order_pos(train_images[:10])
    del train_labels, test_images, test_labels, argv


if __name__ == '__main__':
    main(sys.argv)
