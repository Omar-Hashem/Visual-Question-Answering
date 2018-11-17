import img_features_mxnet
import img_features_tf

def extract(image_batch, model="MXNet"):
    """ 
        if model == "MXNet" (Default if not assigned)
            Requires batch of 3x224x224 Images
            Requires cells of range 0~225
            Example on how to preprocess the image using skimago.io

            def get_image(file_name):
                img = skimage.io.imread(file_name)
                img = transform.resize(img, (224, 224))
                img = np.swapaxes(img, 0, 2)
                img = np.swapaxes(img, 1, 2)
                return np.asarray(img) * 255.0

            img_features.extract([get_image('dog'), get_image('cat')])

            NOTE: for best performance, keep the batch size ALWAYS the SAME or use initialize_graph(max_batch_size)

        if model == "Tensorflow"
            Requires batch of 224x224x3 Images
            Requires cells of range 0~1
            Example on how to preprocess the image using skimago.io

            def get_image(file_name):
                img = skimage.io.imread(file_name)
                img = transform.resize(img, (224, 224))
                return np.asarray(img) 

            img_features.extract([get_image('dog'), get_image('cat')])
    """
    if model == "MXNet":
        if img_features_mxnet._BATCH_SIZE != -1 and img_features_mxnet._BATCH_SIZE != len(image_batch):
            img_features_mxnet.change_batch_size(len(image_batch))
        my_model = img_features_mxnet
    elif model == "Tensorflow":
        my_model = img_features_tf

    return my_model.get_features(image_batch)

def initialize_graph(max_batch_size):
    """
        Accepts a number, represents maximum batch size the feature extractor will use
        Reserves the VGARAM required for the MXNET graph to avoid any future unexpected VGARAM Usage

        initialize_graph(5) will bind the graph for batch_size = 5, 4, 3, 2, 1

        Typically called once, Before extracting any image features
        However, Calling it is optional and NOT required

        ONLY USED WITH MXNET MODEL
    """
    if not img_features_mxnet._IS_CREATED:
        img_features_mxnet._create_model(max_batch_size)
        max_batch_size = max_batch_size - 1

    for i in range(max_batch_size, 0, -1):
        img_features_mxnet.change_batch_size(i)
