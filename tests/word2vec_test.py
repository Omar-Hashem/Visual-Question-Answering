import tests_basis
import sys
from word_preprocess import word2vec, _unload_model

def test_fn(sentence):
    return word2vec(sentence).shape

def main(starting_counter):
    test_args = []

    test_args.append("wow")

    # Should take less time
    test_args.append("ok")

    # Does it cach data?
    test_args.append("ok")

    test_args.append("hello")

    tests_basis.create_tests([test_fn] * len(test_args), test_args, [(300, )] * len(test_args))
    ret = tests_basis.main_tester("Testing The Word2Vector Conversion", starting_counter)

    _unload_model()

    return ret


if __name__ == "__main__":
    tests_basis.set_options(sys.argv)
    main(1)
    