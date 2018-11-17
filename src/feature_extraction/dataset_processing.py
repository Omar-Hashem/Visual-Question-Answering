import sys
sys.path.insert(0, '../data_fetching')
sys.path.insert(0, '../')
from data_fetching import data_path
from data_fetching.img_fetcher import _get_img_by_filename 
import img_features
import pickle 
import os
import glob 

def _get_all_directory_paths(): 
    images_directory_paths = []
    features_directory_paths = []

    images_directory_paths.append(data_path.get_path('training', 'balanced_real_images', 'images'))
    images_directory_paths.append(data_path.get_path('training', 'balanced_binary_abstract_scenes', 'images'))
    images_directory_paths.append(data_path.get_path('training', 'abstract_scenes_v1', 'images'))

    images_directory_paths.append(data_path.get_path('validation', 'balanced_real_images', 'images'))
    images_directory_paths.append(data_path.get_path('validation', 'balanced_binary_abstract_scenes', 'images'))
    images_directory_paths.append(data_path.get_path('validation', 'abstract_scenes_v1', 'images'))

    features_directory_paths.append(data_path.get_path('training', 'balanced_real_images', 'images_features'))
    features_directory_paths.append(data_path.get_path('training', 'balanced_binary_abstract_scenes', 'images_features'))
    features_directory_paths.append(data_path.get_path('training', 'abstract_scenes_v1', 'images_features'))

    features_directory_paths.append(data_path.get_path('validation', 'balanced_real_images', 'images_features'))
    features_directory_paths.append(data_path.get_path('validation', 'balanced_binary_abstract_scenes', 'images_features'))
    features_directory_paths.append(data_path.get_path('validation', 'abstract_scenes_v1', 'images_features'))

    return images_directory_paths, features_directory_paths

def _extract_one_directory_features(batch_size, images_names_list, features_names_list, all_images_len, extracted_counter):

    directory_len = len(images_names_list)

    for i in range(0, directory_len, batch_size):
        images = []
        if directory_len - i < batch_size:
            batch_size = directory_len - i

        for j in range(i, i + batch_size):
            images.append(_get_img_by_filename(images_names_list[j]))

        features = img_features.extract(images)
        _save_features(features, features_names_list[i:i + batch_size])

        extracted_counter = extracted_counter + batch_size

        extra_spaces = len('{}'.format(all_images_len)) - len('{}'.format(extracted_counter))
        prefix = "Extracted ... " + ' ' * extra_spaces
        print(prefix + "{}/{} =~ {:7.3f}%".format(extracted_counter, all_images_len, (extracted_counter / all_images_len) * 100.0))

def _save_features(images_features, features_names_list):

    for i in range(len(images_features)):
            with open(features_names_list[i], 'wb') as fp:
                pickle.dump(images_features[i], fp)

def _get_image_name(image_path):
    idx = -1
    for i in range(len(image_path) - 1, -1, -1):
        if image_path[i] == '/' or image_path[i] == '\\':
            idx = i 
            break 

    temp = image_path[idx + 1:]

    for i in range(len(temp) - 1, 0, -1):
        if temp[i] == '.':
            idx = i 
            break 

    temp = temp[:idx]

    idx = 0
    for i in range(len(temp)):
        if temp[i].isdigit():
            idx = i 
            break

    return temp[idx:]

def _convert_image_path_to_features_path(image_path, features_directory):
    image_name = _get_image_name(image_path)
    return os.path.join(features_directory, image_name + '.bin')

def _convert_image_path_to_features_path_per_directory(images_paths, features_directory):
    features_paths = [] 
    for image_path in images_paths:
        features_paths.append(_convert_image_path_to_features_path(image_path, features_directory))

    return features_paths


def extract_all_features(batch_size):
    images_directory_paths, features_directory_paths = _get_all_directory_paths()

    for directory in features_directory_paths:
        if not os.path.exists(directory):
            os.makedirs(directory)

    all_images_len = 0
    images_path = [] 
    for directory in images_directory_paths:
        images_path.append(glob.glob(directory + "*"))
        all_images_len = all_images_len + len(images_path[-1])

    features_paths = [] 
    for i in range(len(features_directory_paths)):
        features_paths.append(_convert_image_path_to_features_path_per_directory(images_path[i], features_directory_paths[i]))

    extracted_counter = 0
    for i in range(len(images_directory_paths)):
        _extract_one_directory_features(batch_size, images_path[i], features_paths[i], all_images_len, extracted_counter)
        extracted_counter = extracted_counter + len(images_path[i])
