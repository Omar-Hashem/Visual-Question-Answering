import mxnet as mx
from collections import namedtuple
from model_path import get_path

_IS_CREATED = False
_RES_NET_MODEL = None
_SYM = None
_ARG_PARAMS = None
_AUX_PARAMS = None
_BATCH_SIZE = -1

def _create_model(batch_size):
    global _IS_CREATED, _RES_NET_MODEL, _ARG_PARAMS, _AUX_PARAMS, _BATCH_SIZE

    _SYM, _ARG_PARAMS, _AUX_PARAMS = mx.model.load_checkpoint(get_path('VGG19-1k-mxnet'), 0)
    features_layer = _SYM.get_internals()['fc7_output']
    _RES_NET_MODEL = mx.mod.Module(symbol=features_layer, label_names=None, context=mx.gpu())
    _BATCH_SIZE = batch_size
    _bind_model()

    _IS_CREATED = True

def _bind_model():
    _RES_NET_MODEL.bind(for_training=False, data_shapes=[('data', (_BATCH_SIZE, 3, 224, 224))], force_rebind=True) 
    _RES_NET_MODEL.set_params(_ARG_PARAMS, _AUX_PARAMS)

# STILL DOESN'T WORK
def change_batch_size(batch_size):
    global _BATCH_SIZE
    _BATCH_SIZE = batch_size
    _bind_model()

# accepts list of images, each image of 3x224x224
# check tests/extract_features_mxnet.py to understand required preprocessing
def get_features(img_batch):
    if not _IS_CREATED:
        _create_model(len(img_batch))

    Batch = namedtuple('Batch', ['data'])
    _RES_NET_MODEL.forward(Batch([mx.nd.array(img_batch)]))

    return _RES_NET_MODEL.get_outputs()[0].asnumpy()
