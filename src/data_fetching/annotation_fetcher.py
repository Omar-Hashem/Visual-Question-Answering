import json
import operator
from data_fetching.data_path import get_path, get_top_answers_path
import pickle
import os.path

TOP_ANSWERS_PATH = get_top_answers_path()
TOP_ANSWERS_MAP = None      # Key is answer string, Value is its index in the TOP_ANSWERS array
TOP_ANSWERS_LIST = None
TOP_ANSWERS_COUNT = 1000
CLASS_WEIGHT = None 

LOADED_JSON_FILES = {}

# Loads the json file and saves it in the global variable as a dictionary of
# Key = question_id and
# Value = list of answers
def _load_json_file(file_name):

    a_dict = {}
    multiple_choice_answers = {}

    with open(file_name) as data_file:
        data = json.load(data_file)
        annotations = data["annotations"]

    for elem in annotations:
        a_dict[elem["question_id"]] = [(answer["answer"],
                                        answer["answer_confidence"] if "answer_confidence" in answer else "yes") for answer in elem["answers"]]

        multiple_choice_answers[elem["question_id"]] = elem["multiple_choice_answer"]

    # if mc_answer_phase:
    #      return multiple_choice_answers
    # else:
    return a_dict, multiple_choice_answers


# Returns the global dictionary for annotations based on the type of the data set
def _get_annotations(file_name):

    global LOADED_JSON_FILES

    if file_name in LOADED_JSON_FILES:
        return LOADED_JSON_FILES[file_name]
    else:
        LOADED_JSON_FILES[file_name] = _load_json_file(file_name)
        return LOADED_JSON_FILES[file_name]


# Returns a batch of the annotations given the start id of the image, batch size and the type of the data set
# The shape of the returned numpy array is (batch_size, 3, 10)
# 3 is the number of questions for every image
# 10 is the number of answers for each question
def get_annotations_batch(question_ids, file_name):

    # get_annotations()[1] will return a dictionary of question id and multiple choice answer for this question
    all_annotation = _get_annotations(file_name)[1]
    batch = {}

    for q_id in question_ids:
        batch[q_id] = expand_answer(all_annotation[q_id])

    return batch


# Takes the multiple_choice_answer return one hot encoded vector of length TOP_ANSWERS_COUNT
def expand_answer(multiple_choice_answer):

    # TOP_ANSWERS_MAP is a map of the answer word and its index in TOP_ANSWERS_LIST
    global TOP_ANSWERS_MAP, TOP_ANSWERS_LIST

    if TOP_ANSWERS_MAP is None:
        TOP_ANSWERS_MAP, TOP_ANSWERS_LIST, _ = get_top_answers_map()

    expanded_answer = [0] * TOP_ANSWERS_COUNT

    # if the multiple_choice_answer of the question is not in the top answers, then the 
    # return expanded answer will contains all zeroe
    weight = 0
    if multiple_choice_answer in TOP_ANSWERS_MAP:
        expanded_answer[TOP_ANSWERS_MAP[multiple_choice_answer]] = 1
        weight = CLASS_WEIGHT[TOP_ANSWERS_MAP[multiple_choice_answer]]

    return (expanded_answer, weight)


def get_top_answers():
    global TOP_ANSWERS_MAP
    if TOP_ANSWERS_MAP is None:
        TOP_ANSWERS_MAP, TOP_ANSWERS_LIST, _ = get_top_answers_map()

    return TOP_ANSWERS_LIST

# Returns the top answers 
def get_top_answers_map():

    global TOP_ANSWERS_MAP, TOP_ANSWERS_LIST, CLASS_WEIGHT

    if os.path.exists(TOP_ANSWERS_PATH):

        with open(TOP_ANSWERS_PATH, 'rb') as fp:
            TOP_ANSWERS_MAP, TOP_ANSWERS_LIST, CLASS_WEIGHT = pickle.load(fp)

        return TOP_ANSWERS_MAP, TOP_ANSWERS_LIST, CLASS_WEIGHT

    top_answers_dict = {}

    # get_annotations()[1] will return a dictionary of question id and multiple choice answer for this question
    annotations_abstract_v1 = _get_annotations(get_path('training', 'abstract_scenes_v1', 'annotations'))[1]
    annotations_balanced_binary_abstract = _get_annotations(get_path('training', 'balanced_binary_abstract_scenes', 'annotations'))[1]
    annotations_balanced_real = _get_annotations(get_path('training', 'balanced_real_images', 'annotations'))[1]

    all_annotations = [annotations_abstract_v1, annotations_balanced_binary_abstract, annotations_balanced_real]

    for annot_dict in all_annotations:
        for key, multiple_choice_answer in annot_dict.items():

            if multiple_choice_answer in top_answers_dict:
                top_answers_dict[multiple_choice_answer] += 1
            else:
                top_answers_dict[multiple_choice_answer] = 1
               
    # return 2 columns array sorted on the second column, the first column is the answer and the second column is the count
    sorted_top_answers = sorted(top_answers_dict.items(), key=operator.itemgetter(1), reverse=True) 
    # return the first column of the sorted_top_answers, that are the words

    if TOP_ANSWERS_COUNT > len(sorted_top_answers):
        raise ValueError("Top answers count is more than the number of answers !\n TOP_ANSWERS_COUNT = " + TOP_ANSWERS_COUNT)

    TOP_ANSWERS_LIST = ([row[0] for row in sorted_top_answers[:TOP_ANSWERS_COUNT]])

    TOP_ANSWERS_MAP = {}

    for i in range(len(TOP_ANSWERS_LIST)):
        ans = TOP_ANSWERS_LIST[i]
        TOP_ANSWERS_MAP[ans] = i

    CLASS_WEIGHT = _calculate_class_weights(TOP_ANSWERS_LIST, top_answers_dict)

    # Write map and list to file
    with open(TOP_ANSWERS_PATH, 'wb') as fp:
        pickle.dump([TOP_ANSWERS_MAP, TOP_ANSWERS_LIST, CLASS_WEIGHT], fp)

    return TOP_ANSWERS_MAP, TOP_ANSWERS_LIST, CLASS_WEIGHT

def _calculate_class_weights(top_answers_list, top_answers_dict):
    class_weights = [0] * TOP_ANSWERS_COUNT

    for i in range(TOP_ANSWERS_COUNT):
        class_weights[i] = top_answers_dict[top_answers_list[0]] / top_answers_dict[top_answers_list[i]]

    return class_weights

