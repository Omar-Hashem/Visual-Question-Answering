# from tests import tester
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/data_fetching')
sys.path.insert(0, 'src/feature_extraction')

from src.model import evaluate
from src.model import train_model
from src.model import validation_independent
from src.model import trace_statistics
from src.data_fetching.data_fetcher import DataFetcher
from src.feature_extraction.img_features import extract
from src.data_fetching.annotation_fetcher import get_top_answers
from src.sentence_preprocess import question_batch_to_vecs
from src.utility import get_image
from src.feature_extraction.dataset_processing import extract_all_features

def run_tests(system_args):
    # tester.run_tests(system_args)
    pass

def train(batch_size, from_scratch_flag, validate_flag, trace_flag, checkpoint_itr):
    learning_rate = 1e-2
    
    train_model(checkpoint_itr, learning_rate, batch_size, from_scratch_flag, validate_flag, trace_flag)

def evaluate_example_url(image_url, question):
    return evaluate_example(get_image(image_url))

def evaluate_example(image, question):
    image_features = extract(image)
    question_features, words_count = question_batch_to_vecs([question])
    evaluation_logits = evaluate(image_features, question_features, words_count)
    answer_index = evaluation_logits.index(max(evaluation_logits))
    top_answers = get_top_answers()
    return top_answers[answer_index]

def terminate_evaluation():
    pass

def validate_system(batch_size):
    validation_independent(batch_size)

def test_model(batch_size, data_path, model_name, test_size):
    pass

def prepare_data():
    train_loader = DataFetcher('training', preprocessing=True)
    train_loader.preprocess_questions()

    val_loader = DataFetcher('validation', preprocessing=True)
    val_loader.preprocess_questions()

def extract_features(batch_size):
    extract_all_features(batch_size)

def trace(batch_size, k):
    trace_statistics(batch_size, k)
