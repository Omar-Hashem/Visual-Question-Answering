import sys
import os
import traceback
import warnings
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

sys.path.insert(0, '../..')
sys.path.insert(0, '../src')
sys.path.insert(0, '../src/feature_extraction')
sys.path.insert(0, '../src/data_fetching')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tests_path = os.path.dirname(os.path.realpath(__file__))
VQA_Path = os.path.abspath(os.path.join(tests_path, os.pardir))
data_path = os.path.join(VQA_Path, "data")
tests_data_path = os.path.join(data_path, "tests")

TEST_SUCCESS = 0
TEST_ERROR = 1
TEST_FAIL = 2

_FULL_TRACE = False

_testes_files_list = []
_functions = []
_arguments = []
_expected_results = []

def set_options(cmd_variables):
    global _FULL_TRACE

    for arg in cmd_variables:
        if arg == "-f":
            _FULL_TRACE = True
        elif arg == "-tf":
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        elif arg == "-w":
            warnings.filterwarnings("default")
            
def get_test_image_path(file_name):
    return os.path.join(tests_data_path, file_name)

def test(test_fn, args=None, expected_output=None):
    try:
        start, interval = None, None
        if isinstance(args, tuple):
            start = time.time()
            actual_output = test_fn(*args)
        elif args is not None:
            start = time.time()
            actual_output = test_fn(args)
        else:
            start = time.time()
            actual_output = test_fn()

        interval = time.time() - start

        if expected_output == actual_output:
            return TEST_SUCCESS, make_time(interval), None
        else:
            return TEST_ERROR, actual_output, expected_output

    except Exception as e:
        err = None

        if _FULL_TRACE:
            err = traceback.format_exc()
        else:
            err = str(e)

        return TEST_FAIL, err, None

def create_tests(fns, args, exps):
    global _functions, _arguments, _expected_results
    _functions = fns
    _arguments = args
    _expected_results = exps


_GLOBAL_LINE_LEN = 100

def main_tester(test_name, starting_count):
    line_len = _GLOBAL_LINE_LEN

    print('*' * line_len)
    header_1 = '* Testing: {}'.format(test_name)
    space_len_1 = line_len - len(header_1) - 1
    header_2 = '* Number of tests: {}'.format(len(_functions))
    space_len_2 = line_len - len(header_2) - 1
    print(header_1, ' ' * space_len_1, '*', sep='')
    print(header_2, ' ' * space_len_2, '*', sep='')
    print('*' * line_len)

    success, error, fail = 0, 0, 0

    for i in range(len(_functions)):
        test_num = 'Test({})'.format(starting_count)

        print(test_num, sep='', end='')
        sys.stdout.flush()

        t = test(_functions[i], _arguments[i], _expected_results[i])

        if t[0] == TEST_SUCCESS:
            dot_len = line_len - len(test_num) - len(t[1]) - 6
            print('.' * dot_len, t[1], "[PASS]", sep='')
            success = success + 1 
        elif t[0] == TEST_ERROR:
            dot_len = line_len - len(test_num) - 7
            print('.' * dot_len, "[ERROR]", sep='')
            print(" " * (len(test_num) - 1), "Excpected:", t[2])
            print(" " * (len(test_num) - 1), "Found:", t[1])
            error = error + 1
        else: 
            dot_len = line_len - len(test_num) - 6
            print('.' * dot_len, "[FAIL]", sep='')
            print(t[1])
            fail = fail + 1

        starting_count = starting_count + 1

    print()
    print("PASS:  ", success, "/", len(_functions), sep='', )
    print("ERROR: ", error, "/", len(_functions), sep='')
    print("FAIL:  ", fail, "/", len(_functions), sep='')
    print()

    return len(_functions), success, error, fail

def add_test_file(test_file):
    _testes_files_list.append(test_file)


def _print_row(a, b, c, d, e, len_a, len_b=15, len_c=15, len_d=15, len_e=15, first=False):
    if first:
        print('+', '-' * (len_a - 2), '+', '-' * (len_b - 1), '+', '-' * (len_c - 1), '+', '-' * (len_d - 1), '+', '-' * (len_e - 1), '+', sep='')

    print('|', a, ' ' * (len_a - len(a) - 2), '|', sep='', end='')
    print(b, ' ' * (len_b - len(b) - 1), '|', sep='', end='')
    print(c, ' ' * (len_c - len(c) - 1), '|', sep='', end='')
    print(d, ' ' * (len_d - len(d) - 1), '|', sep='', end='')
    print(e, ' ' * (len_e - len(e) - 1), '|', sep='')

    print('+', '-' * (len_a - 2), '+', '-' * (len_b - 1), '+', '-' * (len_c - 1), '+', '-' * (len_d - 1), '+', '-' * (len_e - 1), '+', sep='')


def run_tests():
    line_len = _GLOBAL_LINE_LEN
    n_sum, s_sum, e_sum, f_sum = 0, 0, 0, 0
    nn, ss, ee, ff = [], [], [], []

    for test_file in _testes_files_list:
        n, s, e, f = test_file.main(n_sum + 1)

        nn.append('{}'.format(n))
        ss.append('{}'.format(s))
        ee.append('{}'.format(e))
        ff.append('{}'.format(f))

        n_sum += n
        s_sum += s
        e_sum += e
        f_sum += f

    _print_row("Test File", "#Tests", "Pass", "Error", "Fail", line_len - 60, first=True)

    i = 0
    for test_file in _testes_files_list:
        _print_row(test_file.__name__, nn[i], ss[i], ee[i], ff[i], line_len - 60)
        i = i + 1

    if n_sum == s_sum:
        print("All Tests passed successfully !")

def make_time(interval, length=20):
    r1, r2 = None, None
    if interval < 10**-6:
        r1 = '[%.2f' % (interval * 10**9)
        r2 = 'Nano Secs]'
    elif interval < 10**-3:
        r1 = '[%.2f' % (interval * 10**6)
        r2 = 'Micro Secs]'
    elif interval < 1:
        r1 = '[%.2f' % (interval * 10**3)
        r2 = 'Milli Secs]'
    else:
        r1 = '[%.2f' % interval
        r2 = 'Secs]'

    spaces = ' ' * (length - len(r1) - len(r2))
    ret = r1 + spaces + r2

    return ret 
