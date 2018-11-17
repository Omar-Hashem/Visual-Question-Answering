import tests_basis
import sys
from model import _accuracy
import tensorflow as tf

def test_fn(predictions, labels):
    with tf.Session():
        return int(_accuracy(predictions, labels, k=5, name="Test").eval())

def main(starting_counter):
    test_args, test_exps = [], []

    predictions = tf.constant([[0, 0.1, 0.2, 0.25, 0, 0.1, 0.35], [0, 0.1, 0.2, 0.25, 0, 0.1, 0.35]])
    labels = tf.constant([[0, 0.1, 0, 0.1, 0.1, 0.1, 0.1], [0.2, 0.1, 0, 0, 0.1, 0.1, 0.1]])
                         
    test_args.append((predictions, labels))
    test_exps.append(35)

    tests_basis.create_tests([test_fn] * len(test_args), test_args, test_exps)
    return tests_basis.main_tester("Testing The Top-5 Accuracy", starting_counter)


if __name__ == "__main__":
    tests_basis.set_options(sys.argv)
    main(1)


























