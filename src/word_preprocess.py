from gensim.models.keyedvectors import KeyedVectors
import pickle
from data_fetching.data_path import get_word2vec_model_path, get_glove_path
import os

_WORD2VEC_MODEL = None

def _load_model():
    global _WORD2VEC_MODEL

    if os.path.exists(get_word2vec_model_path()):
        with open(get_word2vec_model_path(), 'rb') as fp:
            _WORD2VEC_MODEL = pickle.load(fp)
    else:
        _WORD2VEC_MODEL = KeyedVectors.load_word2vec_format(get_glove_path(), binary=False)

        with open(get_word2vec_model_path(), 'wb') as fp:
            pickle.dump(_WORD2VEC_MODEL, fp)

def _unload_model():
    global _WORD2VEC_MODEL
    del _WORD2VEC_MODEL
    _WORD2VEC_MODEL = None

def word2vec(word):
    if _WORD2VEC_MODEL is None:
        _load_model()

    if word in _WORD2VEC_MODEL:
        return _WORD2VEC_MODEL[word]
    else:
        return None
