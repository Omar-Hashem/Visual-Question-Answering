import os

feature_extraction_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.abspath(os.path.join(feature_extraction_path, os.pardir))
VQA_path = os.path.abspath(os.path.join(src_path, os.pardir))
models_path = os.path.join(VQA_path, "models")

def get_path(model_name, file_type=""):
    if model_name == "resnet152-1k-tf":
        path = os.path.join(models_path, "resnet152-1k-tf")
        if file_type == "graph":
            return path + "/ResNet-L152.meta"
        elif file_type == "parameters":
            return path + "/ResNet-L152.ckpt"
    elif model_name == "resnet152-11k-mxnet":
        path = os.path.join(models_path, "resnet152-11k-mxnet")
        return path + "/resnet-152"
    elif model_name == "resnet200-1k-mxnet":
        path = os.path.join(models_path, "resnet200-1k-mxnet")
        return path + "/resnet-200"
    elif model_name == "VGG19-1k-mxnet":
        path = os.path.join(models_path, "VGG19-1k-mxnet")
        return path + "/vgg19"
        