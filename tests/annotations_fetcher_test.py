import tests_basis
import sys
from data_fetching.annotation_fetcher import get_annotations_batch
from data_fetching.data_path import get_path


def test_fn(args):
    batch = get_annotations_batch(args[0], args[1])
    return len(batch)


def main(starting_counter):
    test_args, test_exps = [], []

    path = get_path('validation', 'abstract_scenes_v1', 'annotations')

    question_ids = [275780, 275781, 275782, 255060, 255061, 255062]

    test_args.append([question_ids, path])
    test_exps.append(len(question_ids))

    test_args.append([question_ids[:1], path])
    test_exps.append(1)

    test_args.append([question_ids[:5], path])
    test_exps.append(5)

    tests_basis.create_tests([test_fn] * len(test_args), test_args, test_exps)
    return tests_basis.main_tester("Testing annotations batch loading", starting_counter)

if __name__ == "__main__":
    tests_basis.set_options(sys.argv)
    main(1)
