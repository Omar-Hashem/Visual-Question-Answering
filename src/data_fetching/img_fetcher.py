import skimage.io
from skimage import transform
import numpy as np
import glob
import pickle

IMG_SHAPE = (224, 224)  # Used image shape for resizing
IMG_PER_THREAD = 8  # Number of images loaded per thread determined by trial and error according to the batch size

# Returns resized numpy array of the image with this id
def _get_img_by_id(img_dir, img_id):
    suffix = "*" + format(img_id, '012d') + ".*"
    files = glob.glob(img_dir + suffix)

    if len(files) == 0:
        raise ValueError("No image found with suffix = " + suffix)

    return _get_img_by_filename(files[0])

def _get_img_by_filename(file_name, model="MXNet"):
    img = skimage.io.imread(file_name)
    img = transform.resize(img, IMG_SHAPE)

    # Modifications to the image if it's a grayscale or contains an alpha channel
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    mul = 1.0
    if model == "MXNet":
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        mul = 255.0

    return np.asarray(img) * mul

def _get_img_feature_by_id(img_dir, img_id):
    file_name = '2014_' + format(img_id, '012d') + ".bin"
    file_path = img_dir + file_name

    try:
        with open(file_path, 'rb') as fp:
            features = pickle.load(fp)
    except:     
        raise ValueError("No image feature found with name = " + file_name, "in directory:", img_dir)

    return features

# Returns a numpy array containing images and a boolean which is true if we reached the end of the data set
def _get_imgs_batch(img_dir, image_ids):

    batch = {}

    for id in image_ids:
        img = _get_img_by_id(img_dir, id)
        batch[id] = img

    return batch

def _get_imgs_feature_batch(img_dir, image_ids):

    batch = {}

    for id in image_ids:
        img = _get_img_feature_by_id(img_dir, id)
        batch[id] = img

    return batch

# Overloaded for data set type and multi-threading
def get_imgs_batch(image_ids, img_dir):

    # num_threads = math.ceil(len(image_ids) / IMG_PER_THREAD)
    # img_threads = []

    # for i in range(0, num_threads):

    #     ids_slice = image_ids[0: min(IMG_PER_THREAD, len(image_ids))]
    #     image_ids = image_ids[len(ids_slice):]

    #     img_threads.append(FuncThread(_get_imgs_batch, img_dir, ids_slice))

    # batch = {}

    # for i in range(0, num_threads):
    #     batch = {**batch, **img_threads[i].get_ret_val()}

    return _get_imgs_batch(img_dir, image_ids)

# Return features of an image
def get_imgs_features_batch(image_ids, img_dir):

    return _get_imgs_feature_batch(img_dir, image_ids)
    