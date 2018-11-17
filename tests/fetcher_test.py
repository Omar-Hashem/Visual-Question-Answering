import tests_basis
import sys
from data_fetching.data_fetcher import DataFetcher

def test_fn(args):
    val_fetcher = DataFetcher('validation', batch_size=args[0], start_itr=args[1])

    images, questions_vecs, questions_length, annotations, eof = val_fetcher.get_next_batch()

    return [images.shape, questions_vecs.shape, annotations.shape, eof]


def main(starting_counter):
    test_args, test_exps = [], []

    test_args.append([12, 0, True])
    test_exps.append([(12, 2048), (12,30, 300), (12, 1000), False])

    test_args.append([12, 29994, True])
    test_exps.append([(12, 2048), (12,30, 300), (12, 1000), False])
    
    test_args.append([96, 0, True])
    test_exps.append([(96, 2048), (96,30, 300), (96, 1000), False])

    tests_basis.create_tests([test_fn] * len(test_args), test_args, test_exps)
    return tests_basis.main_tester("Testing image batch loading", starting_counter)


if __name__ == "__main__":
    tests_basis.set_options(sys.argv)
    main(1)