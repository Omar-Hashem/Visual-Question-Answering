import tests_basis
import sys
from data_fetching.question_fetcher import get_questions_batch
from data_fetching.data_path import get_path


def test_fn(args):
    batch = get_questions_batch(args[0], args[1], args[2])
    return len(batch)


def main(starting_counter):
    test_args, test_exps = [], []

    path = get_path('validation', 'abstract_scenes_v1', 'questions')

    test_args.append([29994, 6, path])
    test_exps.append(6)

    test_args.append([29900, 32, path])
    test_exps.append(32)

    test_args.append([20000, 20, path])
    test_exps.append(20)

    tests_basis.create_tests([test_fn] * len(test_args), test_args, test_exps)

    return tests_basis.main_tester("Testing questions batch loading", starting_counter)


if __name__ == "__main__":
    tests_basis.set_options(sys.argv)
    main(1)
