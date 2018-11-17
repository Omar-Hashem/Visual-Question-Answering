import tests_basis
import sys
from feature_extraction.dataset_processing import _get_image_name, _convert_image_path_to_features_path

def test_fn_1(image_path):
    return _get_image_name(image_path)

def test_fn_2(image_path, features_directotry):
    return _convert_image_path_to_features_path(image_path, features_directotry)


def main(starting_counter):
    test_args, test_exps, test_fns = [], [], []

    test_fns.append(test_fn_1)
    test_args.append('hello1515.jpg')
    test_exps.append('1515')

    test_fns.append(test_fn_1)
    test_args.append('\hello1515.jpg')
    test_exps.append('1515')

    test_fns.append(test_fn_1)
    test_args.append('\kk\hello1515.jpg')
    test_exps.append('1515')

    test_fns.append(test_fn_2)
    test_args.append(('1515.jpg', 'features_directotry'))
    test_exps.append('features_directotry\\1515.bin')

    test_fns.append(test_fn_2)
    test_args.append(('\hello1515.jpg', 'features_directotry'))
    test_exps.append('features_directotry\\1515.bin')

    test_fns.append(test_fn_2)
    test_args.append(('\kk\hello1515.jpg', 'features_directotry'))
    test_exps.append('features_directotry\\1515.bin')

    tests_basis.create_tests(test_fns, test_args, test_exps)
    return tests_basis.main_tester("Testing some functionality of Dataset feature extraction", starting_counter)


if __name__ == "__main__":
    tests_basis.set_options(sys.argv)
    main(1)
    