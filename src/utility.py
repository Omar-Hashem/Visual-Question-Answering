import numpy as np
import skimage
from skimage import transform

def get_image(file_name, model="MXNet"):

    img = skimage.io.imread(file_name)
    img = transform.resize(img, (224, 224))

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
    