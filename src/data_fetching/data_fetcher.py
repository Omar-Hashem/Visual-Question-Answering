import numpy as np
import random
from data_fetching.img_fetcher import get_imgs_batch, get_imgs_features_batch
from data_fetching.question_fetcher import get_questions_batch, get_questions_len, get_questions
from data_fetching.annotation_fetcher import get_annotations_batch
from data_fetching.data_path import get_path
from sentence_preprocess import question_batch_to_vecs
import pickle
import os
from feature_extraction import img_features


class DataFetcher:

    def __init__(self, evaluation_type, batch_size=32, start_itr=0, preprocessing=False):

        self.evaluation_type = evaluation_type
        self.batch_size = batch_size
        self.itr = start_itr

        self.available_datasets = ['balanced_real_images']

        if not preprocessing:

            self.data_lengthes = [self.get_dataset_len(dataset_name) for dataset_name in self.available_datasets]
            self.sum_data_len = sum(self.data_lengthes)
            self.first_load = False

    #  Returns the name of the current dataset
    def get_current_dataset(self):

        itr = self.itr % self.sum_data_len

        idx = 0
        for length in self.data_lengthes:

            if itr >= length:
                itr -= length
                idx = (idx + 1) % len(self.data_lengthes)

            else:
                break

        return self.available_datasets[idx]

    # Returns the iterator of the current dataset
    def get_dataset_iterator(self):

        itr = self.itr % self.sum_data_len

        for length in self.data_lengthes:

            if itr >= length:
                itr -= length

            else:
                break

        return itr

    # Return path to images of the current dataset
    def get_img_path(self):
        return get_path(self.evaluation_type, self.get_current_dataset(), 'images')

    # Return path to questions of the current dataset
    def get_questions_path(self, dataset_name=None):

        if dataset_name is None:
            dataset_name = self.get_current_dataset()

        return get_path(self.evaluation_type, dataset_name, 'questions')

    # Return path to processed questions
    def get_questions_processed_path(self, dataset_name=None):

        if dataset_name is None:
            dataset_name = self.get_current_dataset()

        path = get_path(self.evaluation_type, dataset_name, 'questions_processed')

        return path

    # Return path to annotations of the current dataset
    def get_annotations_path(self, dataset_name=None):

        if dataset_name is None:
            dataset_name = self.get_current_dataset()

        return get_path(self.evaluation_type, dataset_name, 'annotations')

    def get_img_features_path(self, dataset_name=None):

        if dataset_name is None:
            dataset_name = self.get_current_dataset()

        return get_path(self.evaluation_type, dataset_name, 'images_features')

    # Extract features from images and return a dictionary { "image_id": features }
    def images_to_features(self, images_dict):

        image_ids, batch = list(images_dict.keys()), list(images_dict.values())
        features = img_features.extract(batch)

        for i in range(len(features)):
            images_dict[image_ids[i]] = features[i]

        return images_dict

    # Link questions with images and annotations using ids
    def merge_by_id(self, questions_all, annotations_dict, images_dict):

        annotations, images, weights = [], [], []

        for q in questions_all:

            annotations.append(annotations_dict[q["question_id"]][0])
            weights.append(annotations_dict[q["question_id"]][1])
            images.append(images_dict[q["image_id"]])

        return np.array(annotations), np.array(weights), np.array(images)

    def questions_to_features(self, questions_all):

        questions = [q["question"] for q in questions_all]

        questions_vecs, questions_length = question_batch_to_vecs(questions)

        return questions_vecs, questions_length

    # Load images and use CNN to extract features
    def load_images(self, questions_all):

        # Extract image ids
        image_ids = list(set([elem['image_id'] for elem in questions_all]))

        # Load images
        images_dict = get_imgs_batch(image_ids, self.get_img_path())

        # Extract features from images
        images_dict = self.images_to_features(images_dict)

        return images_dict

    # Load images features
    def load_images_features(self, questions_all):

        # Extract image ids
        image_ids = list(set([elem['image_id'] for elem in questions_all]))

        # Load images
        images_dict = get_imgs_features_batch(image_ids, self.get_img_features_path())

        return images_dict

    def load_annotations(self, questions_all, path=None):

        if path is None:
            path = self.get_annotations_path()

        # Extract question ids
        question_ids = [elem['question_id'] for elem in questions_all]

        # Load annotations
        annotations_dict = get_annotations_batch(question_ids, path)

        return annotations_dict

    # Updates state of the loader to prepare for the next batch
    def update_state(self, actual_batch_size):

        self.first_load = True
        self.itr += actual_batch_size

    def get_dataset_len(self, dataset_name):
        path = self.get_questions_processed_path(dataset_name)
        return get_questions_len(path)

    def _get_next_batch(self, batch_size):

        questions_all = get_questions_batch(self.get_dataset_iterator(), batch_size, self.get_questions_processed_path())

        # Extract features from questions
        # question_features_thread = FuncThread(self.questions_to_features, questions_all)
        
        # Load and extract features from images
        # images_dict_thread = FuncThread(self.load_images, questions_all)

        # Load annotations
        # annotations_dict_thread = FuncThread(self.load_annotations, questions_all)

        # Link questions with images and annotations using ids
        # annotations, images = self.merge_by_id(questions_all, annotations_dict_thread.get_ret_val(), images_dict_thread.get_ret_val())

        annotations, weights, images = self.merge_by_id(questions_all, self.load_annotations(questions_all), 
                                               self.load_images_features(questions_all))

        questions_vecs, questions_length = self.questions_to_features(questions_all)

        # Updates state of the loader to prepare for the next batch
        self.update_state(len(questions_all))

        return images, questions_vecs, questions_length, annotations, weights

    def get_next_batch(self):

        images, questions_vecs, questions_length, annotations, weights = self._get_next_batch(self.batch_size)

        actual_batch_size = len(images)

        eof = self.end_of_data()

        if actual_batch_size < self.batch_size:

            images_2, questions_vecs_2, questions_length_2, annotations_2, weights_2 = self._get_next_batch(self.batch_size - actual_batch_size)

            images = np.append(images, images_2, axis=0)
            questions_vecs = np.append(questions_vecs, questions_vecs_2, axis=0)
            questions_length = np.append(questions_length, questions_length_2, axis=0)
            annotations = np.append(annotations, annotations_2, axis=0)
            weights = np.append(weights, weights_2, axis=0)

        return images, questions_vecs, questions_length, annotations, weights, eof

    def end_of_data(self):
        return (self.itr % self.sum_data_len) == 0 and self.first_load

    # Extract features from images and save them to be used later
    def extract_dataset_images_features(self):

        while(not self.end_of_data()):
            
            questions_batch = get_questions_batch(self.get_dataset_iterator(),
                                                  self.batch_size, self.get_questions_path())

            images_dict = self.load_images(questions_batch)

            self.write_images_features(images_dict)

            self.update_state(len(questions_batch))

            print("PREPROCESSING:", self.itr, "/", self.sum_data_len, " ", (self.itr / self.sum_data_len))

    # Write images features to features subfolder
    def write_images_features(self, images_dict):

        directory = self.get_img_features_path()
        if not os.path.exists(directory):
            os.makedirs(directory)

        for image_id, feature in images_dict.items():

            with open(os.path.join(directory, format(image_id, '012d') + '.bin'), 'wb') as fp:
                pickle.dump(feature, fp)

    # Removes questions with annotations not in the Top Answers
    def preprocess_questions(self):

        for dataset in self.available_datasets:

            questions_all = get_questions(self.get_questions_path(dataset))
            annotations_all = self.load_annotations(questions_all, path=self.get_annotations_path(dataset))

            valid_questions = []

            print("PREPROCESSING: ", self.get_questions_path(dataset))

            for q in questions_all:

                if sum(annotations_all[q["question_id"]][0]) != 0:

                    valid_questions.append(q)

            random.shuffle(valid_questions)

            print("Removed {} questions".format(len(questions_all) - len(valid_questions)))

            with open(self.get_questions_processed_path(dataset), 'wb') as fp:
                    pickle.dump(valid_questions, fp)
