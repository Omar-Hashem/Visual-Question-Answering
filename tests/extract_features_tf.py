import tests_basis
import sys
from feature_extraction.img_features_tf import get_features, _close_session
import skimage.io
from skimage import transform
import numpy as np

def get_image(file_name):
    """ Required Image PreProcessing for the TensorFlow Model """
    img = skimage.io.imread(tests_basis.get_test_image_path(file_name))
    return transform.resize(img, (224, 224))

def test_fn_1():
    img = get_image('resnet_test.jpg')
    batch = np.array([img] * 2)
    return get_features(batch).shape

# Should compute in less time than fn_1
def test_fn_2():
    img = get_image('resnet_test.jpg')
    batch = np.array([img] * 2)
    return get_features(batch).shape

def test_fn_3():
    img = get_image('resnet_test.jpg')
    batch = np.array([img] * 4)
    return get_features(batch).shape

def main(starting_counter):
    test_args, test_exps, test_fns = [], [], []

    test_args = [None] * 3

    test_fns.append(test_fn_1)
    test_exps.append((2, 2048))

    test_fns.append(test_fn_2)
    test_exps.append((2, 2048))

    test_fns.append(test_fn_3)
    test_exps.append((4, 2048))

    tests_basis.create_tests(test_fns, test_args, test_exps)
    ret = tests_basis.main_tester("Testing the feature extraction from the resnet152-1k-tf", starting_counter)

    _close_session()

    return ret


if __name__ == "__main__":
    tests_basis.set_options(sys.argv)
    main(1)
