from collections import namedtuple
from os.path import join, abspath, dirname, pardir

SAMPLING_FREQUENCY = 1500
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')

Animal = namedtuple('Animal', {'directory', 'short_name'})
ANIMALS = {
    'HPa': Animal(directory=join(RAW_DATA_DIR, 'HPa_direct'),
                  short_name='HPa'),
}

REPLAY_COVARIATES = ['session_time', 'replay_task',
                     'replay_order', 'replay_motion']
