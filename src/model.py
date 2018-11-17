import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow import losses
from tensorflow.contrib import rnn
from data_fetching.data_fetcher import DataFetcher
import numpy as np
import os 
import datetime

_MAIN_MODEL_GRAPH = None

def dense_batch(input_ph, phase, output_size, has_dropout=False, has_relu=False, has_bn=False, has_tanh=False, name=None):
    my_layer = tf.contrib.layers.fully_connected(input_ph, output_size, activation_fn=None)

    if has_dropout:
        layers.dropout(my_layer, is_training=phase)
    if has_bn:
        my_layer = tf.contrib.layers.batch_norm(my_layer, is_training=phase)
    if has_relu:
        my_layer = tf.nn.relu(my_layer, name)
    if has_tanh:
        my_layer = tf.nn.tanh(my_layer, name)
    
    return my_layer

# question_ph is batchSize*#wordsInEachQuestion*300
def question_lstm_model(questions_ph, phase_ph, questions_length_ph, cell_size, layers_num):
    
    mcell = rnn.MultiRNNCell([rnn.LSTMCell(cell_size, state_is_tuple=True) for _ in range(layers_num)])

    init_state = mcell.zero_state(tf.shape(questions_ph)[0], tf.float32) 
    _, final_state = tf.nn.dynamic_rnn(mcell, questions_ph, sequence_length=questions_length_ph, initial_state=init_state)
    
    combined_states = tf.stack(final_state, 1)
    combined_states = tf.reshape(combined_states, [-1, cell_size * layers_num * 2])

    return dense_batch(combined_states, phase_ph, 1024, has_tanh=True, has_bn=True)  # The questions features

def abstract_model(questions_ph, img_features_ph, questions_length_ph, phase_ph, cell_size=512, layers_num=2):

    question_features = question_lstm_model(questions_ph, phase_ph, questions_length_ph, cell_size, layers_num)
    img_features = dense_batch(img_features_ph, phase_ph, 1024, has_tanh=True, has_bn=True)
    fused_features = tf.multiply(img_features, question_features)
    
    return dense_batch(fused_features, None, 1000)  # logits

def _accuracy(predictions, labels, k, name):  
    
    _, top_indices = tf.nn.top_k(predictions, k, sorted=True, name=None)

    x = tf.to_int32(tf.shape(top_indices))[0]
    y = tf.to_int32(tf.shape(top_indices))[1]
    flattened_ind = tf.range(0, tf.multiply(x, y)) // y * tf.shape(labels)[1] + tf.reshape(top_indices, [-1])

    acc = tf.reduce_sum(tf.gather(tf.reshape(labels, [-1]), flattened_ind)) / tf.to_float(tf.shape(labels))[0] * 100
    return tf.identity(acc, name)

def save_state(saver, sess, starting_pos, idx, batch_size, loss_sum, accuracy_1_sum, accuracy_5_sum, cnt_iteration, cnt_examples, epoch_number):

    directory = os.path.join(os.getcwd(), "models/VQA_model/")
    if not os.path.exists(directory):
        os.makedirs(directory)

    saver.save(sess, os.path.join(os.getcwd(), "models/VQA_model/main_model"), global_step=starting_pos + idx * batch_size)
    np.savetxt('models/VQA_model/statistics.out', (loss_sum, accuracy_1_sum, accuracy_5_sum, cnt_iteration, cnt_examples, epoch_number))

def validation_independent(batch_size):
    sess = tf.Session()
    questions_place_holder, images_place_holder, labels_place_holder, class_weight_place_holder, questions_length_place_holder, logits, loss, top1_accuarcy, top5_accuarcy, phase_ph, starting_pos, train_step = _get_saved_graph_tensors(sess)

    validation_loss, validation_acc_1, validation_acc_5 = validation_acc_loss(sess,
                                                                              batch_size,
                                                                              images_place_holder,
                                                                              questions_place_holder,
                                                                              labels_place_holder,
                                                                              class_weight_place_holder,
                                                                              questions_length_place_holder,
                                                                              phase_ph, top1_accuarcy, top5_accuarcy, loss)
    
    to_print = "Validation::SUMMARY (Avg-Top1-Accuracy: {}%, Avg-Top5-Accuracy: {}%, Avg-Loss: {})".format(validation_acc_1, validation_acc_5, validation_loss)
    print(to_print)

def trace_statistics(batch_size, k=10):
    sess = tf.Session()
    questions_place_holder, images_place_holder, labels_place_holder, class_weight_place_holder, questions_length_place_holder, logits, loss, top1_accuarcy, top5_accuarcy, phase_ph, starting_pos, train_step = _get_saved_graph_tensors(sess)

    val_data_fetcher = DataFetcher('validation', batch_size=batch_size)
    images_batch, questions_batch, questions_length, labels_batch, weights_batch, end_of_epoch = val_data_fetcher.get_next_batch()

    feed_dict = {questions_place_holder: questions_batch,
                 images_place_holder: images_batch,
                 labels_place_holder: labels_batch,
                 class_weight_place_holder: weights_batch,
                 questions_length_place_holder: questions_length,
                 phase_ph: 0}

    logits_softmax = tf.nn.softmax(logits)
    top_values, top_indices = tf.nn.top_k(logits_softmax, k, sorted=True, name=None)

    val_loss, val_top_values, val_top_indices = sess.run([loss, top_values, top_indices], feed_dict=feed_dict)

    time_stamp = 'time_{:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now())
    file_name = "log/statistics_trace" + "_k{}_".format(k) + time_stamp

    my_file = open(file_name, 'w')
    my_file.write('K={}, Loss = {}\n'.format(k, val_loss))

    labels_max_indices = np.argmax(labels_batch, axis=1)
    for i in range(batch_size):
        my_file.write('Label-index:' + '{}'.format(labels_max_indices[i]).ljust(5))
        for j in range(k):
            my_file.write(' - ' + '{}'.format(val_top_indices[i][j]).rjust(4) + ':{:1.5f}'.format(val_top_values[i][j])) 

        my_file.write('\n')

    my_file.close()



def validation_acc_loss(sess,
                        batch_size,
                        images_place_holder,
                        questions_place_holder,
                        labels_place_holder,
                        class_weight_place_holder,
                        questions_length_place_holder,
                        phase_ph,
                        top1_accuarcy,
                        top5_accuarcy,
                        loss):

    print("VALIDATION:: STARTING...")

    temp_acc_1 = 0.0
    temp_acc_5 = 0.0
    temp_loss = 0.0
    
    itr = 0

    val_data_fetcher = DataFetcher('validation', batch_size=batch_size)

    while True:

        images_batch, questions_batch, questions_length, labels_batch, weights_batch, end_of_epoch = val_data_fetcher.get_next_batch() 
        
        feed_dict = {questions_place_holder: questions_batch,
                     images_place_holder: images_batch,
                     labels_place_holder: labels_batch,
                     class_weight_place_holder: weights_batch,
                     questions_length_place_holder: questions_length,
                     phase_ph: 0}

        l, a1, a5 = sess.run([loss, top1_accuarcy, top5_accuarcy], feed_dict=feed_dict)
        
        itr += 1
        temp_acc_1 += a1
        temp_acc_5 += a5
        temp_loss += l

        print("VALIDATION:: Iteration[{}]".format(itr))

        if(end_of_epoch):
            break
    
    temp_acc_1 /= itr
    temp_acc_5 /= itr
    temp_loss /= itr
    
    print("VALIDATION:: ENDING...")

    return temp_loss, temp_acc_1, temp_acc_5

def train_model(check_point_iteration,
                learning_rate, 
                batch_size,
                from_scratch=False,
                validate=True, 
                trace=False):
                    
    sess = tf.Session()
    
    if from_scratch:
        questions_place_holder, images_place_holder, labels_place_holder, class_weight_place_holder, questions_length_place_holder, logits, loss, top1_accuarcy, top5_accuarcy, phase_ph = _train_from_scratch(sess) 
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=1)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            train_step = optimizer.minimize(loss, name='train_step')

        init = tf.global_variables_initializer()
        sess.run(init)

        starting_pos = 0
        loss_sum, accuracy_1_sum, accuracy_5_sum, cnt_iteration, cnt_examples = 0.0, 0.0, 0.0, 0.0, 0.0
        epoch_number = 1

        training_log_file, training_statistics_file, validation_statistics_file = _create_printing_files(discard_if_exists=True)
    else:
        questions_place_holder, images_place_holder, labels_place_holder, class_weight_place_holder, questions_length_place_holder, logits, loss, top1_accuarcy, top5_accuarcy, phase_ph, starting_pos, train_step = _get_saved_graph_tensors(sess)
        loss_sum, accuracy_1_sum, accuracy_5_sum, cnt_iteration, cnt_examples, epoch_number = np.loadtxt('models/VQA_model/statistics.out')
        training_log_file, training_statistics_file, validation_statistics_file = _create_printing_files(discard_if_exists=False)

    saver = tf.train.Saver(max_to_keep=1)

    train_data_fetcher = DataFetcher('training', batch_size=batch_size, start_itr=starting_pos)

    i = 1
    while True:
        
        images_batch, questions_batch, questions_length, labels_batch, weights_batch, end_of_epoch = train_data_fetcher.get_next_batch()

        if validate and end_of_epoch:
            validation_loss, validation_acc_1, validation_acc_5 = validation_acc_loss(sess,
                                                                                      batch_size,
                                                                                      images_place_holder,
                                                                                      questions_place_holder,
                                                                                      labels_place_holder,
                                                                                      class_weight_place_holder,
                                                                                      questions_length_place_holder,
                                                                                      phase_ph, top1_accuarcy, top5_accuarcy, loss)
            _print_statistics(validation_statistics_file, "Validation", epoch_number, validation_acc_1, validation_acc_5, validation_loss)

        if end_of_epoch: 
            accuracy_1_avg = accuracy_1_sum / cnt_iteration
            accuracy_5_avg = accuracy_5_sum / cnt_iteration
            loss_avg = loss_sum / cnt_iteration
            _print_statistics(training_statistics_file, "Training", epoch_number, accuracy_1_avg, accuracy_5_avg, loss_avg)
            epoch_number = epoch_number + 1
            loss_sum, accuracy_1_sum, accuracy_5_sum, cnt_iteration, cnt_examples = 0.0, 0.0, 0.0, 0.0, 0.0
            save_state(saver, sess, starting_pos, i, batch_size, loss_sum, accuracy_1_sum, accuracy_5_sum, cnt_iteration, cnt_examples, epoch_number)

        feed_dict = {questions_place_holder: questions_batch,
                     images_place_holder: images_batch, 
                     labels_place_holder: labels_batch, 
                     class_weight_place_holder: weights_batch,
                     questions_length_place_holder: questions_length, 
                     phase_ph: 1}
        
        _, training_loss, training_acc_1, training_acc_5 = sess.run([train_step, loss, top1_accuarcy, top5_accuarcy], feed_dict=feed_dict)

        cnt_iteration += 1
        cnt_examples += batch_size
        loss_sum += training_loss
        accuracy_1_sum += training_acc_1
        accuracy_5_sum += training_acc_5

        if i % check_point_iteration == 0 and not end_of_epoch:
            save_state(saver, sess, starting_pos, i, batch_size, loss_sum, accuracy_1_sum, accuracy_5_sum, cnt_iteration, cnt_examples, epoch_number)
        
        if trace:  # trace is only for training log
            _print_training_log(training_log_file, i, epoch_number, training_acc_1, training_acc_5, training_loss)

        i = i + 1
        
    sess.close()

def _print_training_log(file, iteration, epoch, training_acc_1, training_acc_5, training_loss):
    to_print = "TRAINING::Epoch[{}]-Iteration[{}]: (Top1-Accuracy: {}%, Top5-Accuracy: {}%, Loss: {})".format(epoch, iteration, training_acc_1, training_acc_5, training_loss)
    print(to_print)
    file.write(to_print)
    file.write('\n')
    file.flush()

def _print_statistics(file, evaluation_type, epoch, accuracy_1_avg, accuracy_5_avg, loss_avg):
    to_print = evaluation_type + "::SUMMARY::Epoch[{}] (Avg-Top1-Accuracy: {}%, Avg-Top5-Accuracy: {}%, Avg-Loss: {})".format(epoch, accuracy_1_avg, accuracy_5_avg, loss_avg)
    file.write(to_print)
    file.write('\n')
    file.flush()

def _create_printing_files(discard_if_exists=False):
    mode = 'a'
    if discard_if_exists:
        mode = 'w'

    directory = os.path.join(os.getcwd(), "log/")
    if not os.path.exists(directory):
        os.makedirs(directory)

    training_log_file = open('log/training_log.txt', mode)
    training_statistics_file = open('log/training_statistics.txt', mode)
    validation_statistics_file = open('log/validation_statistics.txt', mode)

    return training_log_file, training_statistics_file, validation_statistics_file

def _load_model(sess):
    meta_graph_path, data_path, last_index = _get_last_main_model_path()
    new_saver = tf.train.import_meta_graph(meta_graph_path)

    # requires a session in which the graph was launched.
    new_saver.restore(sess, data_path)
    
    global _MAIN_MODEL_GRAPH
    _MAIN_MODEL_GRAPH = tf.get_default_graph()
    return last_index

def _get_last_main_model_path():
    checkpoint_file = open('./models/VQA_model/checkpoint', 'r')
    meta_graph_path = None
    data_path = None
    lst_indx = 0
    for line in checkpoint_file: 
        final_line = line
    word = None
  
    if final_line is not None:
        strt = final_line.find("main_model", 0)
        if strt != -1:
            word = final_line[strt:len(final_line) - 2]
            strt2 = word.find("-", 0)
            lst_indx = int(word[strt2 + 1:len(word)])
            
    if word is not None:
        meta_graph_path = "./models/VQA_model/" + word + ".meta"
        data_path = "./models/VQA_model/" + word
    return meta_graph_path, data_path, lst_indx

def _train_from_scratch(sess):
    questions_place_holder = tf.placeholder(tf.float32, [None, None, 300], name='questions_place_holder') 
    images_place_holder = tf.placeholder(tf.float32, [None, 4096], name='images_place_holder')
    labels_place_holder = tf.placeholder(tf.float32, [None, 1000], name='labels_place_holder')
    class_weight_place_holder = tf.placeholder(tf.float32, [None], name='class_weight_place_holder')
    questions_length_place_holder = tf.placeholder(tf.int32, [None], name='questions_length_place_holder')

    bn_phase = tf.placeholder(tf.bool, [], name='bn_phase')

    logits = tf.identity(abstract_model(questions_place_holder, images_place_holder, questions_length_place_holder, bn_phase), name="logits")
    loss = tf.identity(losses.softmax_cross_entropy(labels_place_holder, logits), name='loss')
    top1_accuarcy = _accuracy(tf.nn.softmax(logits), labels_place_holder, k=1, name='top1_accuracy')
    top5_accuarcy = _accuracy(tf.nn.softmax(logits), labels_place_holder, k=5, name='top5_accuracy')

    return questions_place_holder, images_place_holder, labels_place_holder, class_weight_place_holder, questions_length_place_holder, logits, loss, top1_accuarcy, top5_accuarcy, bn_phase

def _get_saved_graph_tensors(sess):
    
    last_index = 0
    if _MAIN_MODEL_GRAPH is None:
        last_index = _load_model(sess)
    
    questions_place_holder = _MAIN_MODEL_GRAPH.get_tensor_by_name("questions_place_holder:0") 
    images_place_holder = _MAIN_MODEL_GRAPH.get_tensor_by_name("images_place_holder:0")
    labels_place_holder = _MAIN_MODEL_GRAPH.get_tensor_by_name("labels_place_holder:0")
    class_weight_place_holder = _MAIN_MODEL_GRAPH.get_tensor_by_name("class_weight_place_holder:0")
    questions_length_place_holder = _MAIN_MODEL_GRAPH.get_tensor_by_name("questions_length_place_holder:0")
    
    bn_phase = _MAIN_MODEL_GRAPH.get_tensor_by_name("bn_phase:0")

    logits = _MAIN_MODEL_GRAPH.get_tensor_by_name("logits:0")
    loss = _MAIN_MODEL_GRAPH.get_tensor_by_name("loss:0")
    top1_accuarcy = _MAIN_MODEL_GRAPH.get_tensor_by_name("top1_accuracy:0")
    top5_accuarcy = _MAIN_MODEL_GRAPH.get_tensor_by_name("top5_accuracy:0")

    train_step = _MAIN_MODEL_GRAPH.get_operation_by_name("train_step")

    return questions_place_holder, images_place_holder, labels_place_holder, class_weight_place_holder, questions_length_place_holder, logits, loss, top1_accuarcy, top5_accuarcy, bn_phase, last_index, train_step

def evaluate(image_features, question_features, questions_length):

    sess = tf.Session()
    questions_place_holder, images_place_holder, labels_place_holder, class_weight_place_holder, questions_length_place_holder, logits, _, _, phase_ph = _get_saved_graph_tensors(sess)
    feed_dict = {questions_place_holder: question_features, images_place_holder: image_features, questions_length_place_holder: questions_length, phase_ph: 0}
    
    results = tf.nn.softmax(logits)
    evaluation_logits = sess.run([results], feed_dict=feed_dict)

    return evaluation_logits
