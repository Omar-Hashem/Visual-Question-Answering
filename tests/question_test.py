import tests_basis
import sys
from sentence_preprocess import preprocess

def test_fn(sentence):
    return preprocess(sentence)

def main(starting_counter):
    test_args, test_exps = [], []

    test_args.append("hello world !")
    test_exps.append(["hello", "world"])

    test_args.append("haven't you seen?")
    test_exps.append(["have", "not", "you", "see"])

    test_args.append("ok... I saw him !!")
    test_exps.append(["ok", "i", "saw", "him"])

    test_args.append("it's 5.30 o'clock, come fast")
    test_exps.append(["it", "be", "5.30", "o'clock", "come", "fast"])

    test_args.append("he ran from ahmed.")
    test_exps.append(["he", "run", "from", "ahmed"])

    test_args.append("does it contains sugar?")
    test_exps.append(["do", "it", "contain", "sugar"])

    test_args.append("play football..")
    test_exps.append(["play", "football"])

    test_args.append("OK")
    test_exps.append(["ok"])

    test_args.append("you're a good looking person.")
    test_exps.append(["you", "be", "a", "good", "look", "person"])

    test_args.append("he'd dragged me away")
    test_exps.append(["he", "have", "drag", "me", "away"])

    test_args.append("he'd drag me away")
    test_exps.append(["he", "would", "drag", "me", "away"])

    tests_basis.create_tests([test_fn] * len(test_args), test_args, test_exps)
    return tests_basis.main_tester("Testing The sentence preprocessing", starting_counter)


if __name__ == "__main__":
    tests_basis.set_options(sys.argv)
    main(1)
    