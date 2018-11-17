import os

data_fetching_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.abspath(os.path.join(data_fetching_path, os.pardir))
VQA_path = os.path.abspath(os.path.join(src_path, os.pardir))
data_path = os.path.join(VQA_path, "data")
VQA_Dataset_path = os.path.join(data_path, "VQA_Dataset")
training_path = os.path.join(VQA_Dataset_path, "training")
validation_path = os.path.join(VQA_Dataset_path, "validation")
testing_path = os.path.join(VQA_Dataset_path, "testing")

def get_path(evaluation_type, data_set_type, data_type):

    ret_path = ""

    if evaluation_type == 'training':

        ret_path = training_path
        ret_path = os.path.join(ret_path, data_set_type)

        if data_set_type == 'balanced_real_images':
            if data_type == 'annotations':              
                ret_path = os.path.join(ret_path, 'v2_mscoco_train2014_annotations.json')
            elif data_type == 'questions':
                ret_path = os.path.join(ret_path, 'v2_OpenEnded_mscoco_train2014_questions.json')
            elif data_type == 'questions_processed':
                ret_path = os.path.join(ret_path, 'v2_OpenEnded_mscoco_train2014_questions_processed.bin')
            elif data_type == 'images':
                ret_path = os.path.join(ret_path, 'train2014/')
            elif data_type == 'images_features':
                ret_path = os.path.join(ret_path, 'train2014_features/')
            elif data_type == 'complementary_pairs_list':
                pass

        elif data_set_type == 'balanced_binary_abstract_scenes':
            if data_type == 'annotations':
                ret_path = os.path.join(ret_path, 'abstract_v002_train2017_annotations.json')
            elif data_type == 'questions':
                ret_path = os.path.join(ret_path, 'OpenEnded_abstract_v002_train2017_questions.json')
            elif data_type == 'questions_processed':
                ret_path = os.path.join(ret_path, 'OpenEnded_abstract_v002_train2017_questions_processed.bin')
            elif data_type == 'images':
                ret_path = os.path.join(ret_path, 'scene_img_abstract_v002_binary_train2017/')
            elif data_type == 'images_features':
                ret_path = os.path.join(ret_path, 'scene_img_abstract_v002_train2017_features/')

        elif data_set_type == 'abstract_scenes_v1':  # NEED CHANGE TO TRAINING
            if data_type == 'annotations':
                ret_path = os.path.join(ret_path, 'abstract_v002_train2015_annotations.json')
            elif data_type == 'questions':
                ret_path = os.path.join(ret_path, 'OpenEnded_abstract_v002_train2015_questions.json')
            elif data_type == 'questions_processed':
                ret_path = os.path.join(ret_path, 'OpenEnded_abstract_v002_train2015_questions_processed.bin')
            elif data_type == 'images':
                ret_path = os.path.join(ret_path, 'scene_img_abstract_v002_train2015/')
            elif data_type == 'images_features':
                ret_path = os.path.join(ret_path, 'scene_img_abstract_v002_train2015_features/')

    elif evaluation_type == 'validation':

        ret_path = validation_path
        ret_path = os.path.join(ret_path, data_set_type)

        if data_set_type == 'balanced_real_images':
            if data_type == 'annotations':
                ret_path = os.path.join(ret_path, 'v2_mscoco_val2014_annotations.json')
            elif data_type == 'questions':     
                ret_path = os.path.join(ret_path, 'v2_OpenEnded_mscoco_val2014_questions.json')
            elif data_type == 'questions_processed':     
                ret_path = os.path.join(ret_path, 'v2_OpenEnded_mscoco_val2014_questions_processed.bin')
            elif data_type == 'images':
                ret_path = os.path.join(ret_path, 'val2014/')
            elif data_type == 'images_features':
                ret_path = os.path.join(ret_path, 'val2014_features/')

            elif data_type == 'complementary_pairs_list':
                pass

        elif data_set_type == 'balanced_binary_abstract_scenes':
            if data_type == 'annotations':
                ret_path = os.path.join(ret_path, 'abstract_v002_val2017_annotations.json')
            elif data_type == 'questions':
                ret_path = os.path.join(ret_path, 'OpenEnded_abstract_v002_val2017_questions.json')
            elif data_type == 'questions_processed':
                ret_path = os.path.join(ret_path, 'OpenEnded_abstract_v002_val2017_questions_processed.bin')
            elif data_type == 'images':
                ret_path = os.path.join(ret_path, 'scene_img_abstract_v002_binary_val2017/')
            elif data_type == 'images_features':
                ret_path = os.path.join(ret_path, 'scene_img_abstract_v002_binary_val2017_features/')

        elif data_set_type == 'abstract_scenes_v1':
            if data_type == 'annotations':
                ret_path = os.path.join(ret_path, 'abstract_v002_val2015_annotations.json')
            elif data_type == 'questions':  
                ret_path = os.path.join(ret_path, 'OpenEnded_abstract_v002_val2015_questions.json')
            elif data_type == 'questions_processed':
                ret_path = os.path.join(ret_path, 'OpenEnded_abstract_v002_val2015_questions_processed.bin')
            elif data_type == 'images':    
                ret_path = os.path.join(ret_path, 'scene_img_abstract_v002_val2015/')
            elif data_type == 'images_features':
                ret_path = os.path.join(ret_path, 'scene_img_abstract_v002_val2015_features/')

    elif evaluation_type == 'testing':

        ret_path = testing_path
        ret_path = os.path.join(ret_path, data_set_type)

        if data_set_type == 'balanced_real_images':
            if data_type == 'questions':
                ret_path = os.path.join(ret_path, '7amada.json')
            elif data_type == 'images':
                ret_path = os.path.join(ret_path, 'images/')

        elif data_set_type == 'abstract_scenes_v1':
            if data_type == 'questions':
                ret_path = os.path.join(ret_path, '7amada.json')
            elif data_type == 'images':
                ret_path = os.path.join(ret_path, '7amada.json')

    return ret_path


def get_top_answers_path():
    return os.path.join(data_path, "top_answers.bin")

def get_word2vec_model_path():
    return os.path.join(data_path, "word2vec_model.bin")

def get_glove_path():
    return os.path.join(data_path, "glove.840B.300d.txt")
    