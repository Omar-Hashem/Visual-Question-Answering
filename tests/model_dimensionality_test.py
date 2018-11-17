import tests_basis
import sys
import tensorflow as tf
from model import _train_from_scratch

def test_fn(tensor_type, unique_num):
    with tf.variable_scope('{}'.format(unique_num)):
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        questions_place_holder, images_place_holder, labels_place_holder, class_weight_place_holder, questions_length_place_holder, logits, loss, accuarcy_1, accuracy_5, bn_phase = _train_from_scratch(sess)
        sess.close()

        if tensor_type == "questions_place_holder":
            return questions_place_holder.get_shape().as_list()
        elif tensor_type == "images_place_holder":
            return images_place_holder.get_shape().as_list()
        elif tensor_type == "labels_place_holder":
            return labels_place_holder.get_shape().as_list()
        elif tensor_type == "logits":
            return logits.get_shape().as_list()
        elif tensor_type == "loss":
            return loss.get_shape().as_list()
        elif tensor_type == "bn_phase":
            return bn_phase.get_shape().as_list()
        elif tensor_type == "class_weight":
            return class_weight_place_holder.get_shape().as_list()
        return accuarcy_1.get_shape().as_list()


def main(starting_counter):
    test_args, test_exps = [], []
    
    test_args.append(("questions_place_holder", 1))
    test_exps.append([None, None, 300])
    test_args.append(("images_place_holder", 2))
    test_exps.append([None, 2048])
    test_args.append(("labels_place_holder", 3))
    test_exps.append([None, 1000])
    test_args.append(("logits", 4))
    test_exps.append([None, 1000])
    test_args.append(("loss", 5))
    test_exps.append([])
    test_args.append(("accuracy", 6))
    test_exps.append([])
    test_args.append(("bn_phase", 7))
    test_exps.append([])
    test_args.append(("class_weight", 8))
    test_exps.append([None])

    tests_basis.create_tests([test_fn] * len(test_args), test_args, test_exps)
    return tests_basis.main_tester("Testing The model dimentionality based on Batch_Size", starting_counter)

if __name__ == "__main__":
    tests_basis.set_options(sys.argv)
    main(1)
    