import tests_basis
import sys
from feature_extraction.img_features import extract
import skimage.io
from skimage import transform
import numpy as np

def get_image(file_name):
    """ Required Image PreProcessing for the MXNet Model """
    img = skimage.io.imread(tests_basis.get_test_image_path(file_name))
    img = transform.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    return np.asarray(img) * 255.0

def test_fn_1():
    img = get_image('resnet_test.jpg')
    return extract([img]).shape

def test_fn_2():
    """ Exact code as test_fn_1 but should finish in less time """
    img = get_image('resnet_test.jpg')
    return extract([img]).shape

def test_fn_3():
    img = get_image('resnet_test.jpg')
    return extract([img] * 4).shape

def test_fn_4():
    img = get_image('resnet_test.jpg')
    return extract([img] * 4).shape

def test_fn_5():
    img = get_image('resnet_test.jpg')
    return extract([img] * 5).shape

def main(starting_counter):
    test_args, test_exps, test_fns = [], [], []

    test_args = [None] * 5

    test_fns.append(test_fn_1)
    test_exps.append((1, 2048))

    test_fns.append(test_fn_2)
    test_exps.append((1, 2048))

    test_fns.append(test_fn_3)
    test_exps.append((4, 2048))

    test_fns.append(test_fn_4)
    test_exps.append((4, 2048))

    test_fns.append(test_fn_5)
    test_exps.append((5, 2048))

    tests_basis.create_tests(test_fns, test_args, test_exps)
    return tests_basis.main_tester("Testing the general feature extraction", starting_counter)


if __name__ == "__main__":
    tests_basis.set_options(sys.argv)
    main(1)
    