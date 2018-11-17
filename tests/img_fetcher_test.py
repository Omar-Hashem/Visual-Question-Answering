import tests_basis
import sys
from data_fetching.img_fetcher import get_imgs_batch
from data_fetching.data_path import get_path



def test_fn(args):
    batch = get_imgs_batch(args[0], args[1])
    return len(batch)


def main(starting_counter):
    test_args, test_exps = [], []

    path = get_path('validation', 'abstract_scenes_v1', 'images')

    image_ids = [20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009]

    test_args.append([image_ids, path])
    test_exps.append(len(image_ids))

    test_args.append([image_ids[:1], path])
    test_exps.append(1)

    test_args.append([image_ids[:5], path])
    test_exps.append(5)

    tests_basis.create_tests([test_fn] * len(test_args), test_args, test_exps)
    return tests_basis.main_tester("Testing image batch loading", starting_counter)


if __name__ == "__main__":
    tests_basis.set_options(sys.argv)
    main(1)
