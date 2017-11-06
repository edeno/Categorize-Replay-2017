from collections import namedtuple
from os.path import join, abspath, dirname, pardir

import seaborn as sns

# LFP sampling frequency
SAMPLING_FREQUENCY = 1500

# Data directories and definitions
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')

Animal = namedtuple('Animal', {'directory', 'short_name'})
ANIMALS = {
    'HPa': Animal(directory=join(RAW_DATA_DIR, 'HPa_direct'),
                  short_name='HPa'),
    'HPb': Animal(directory=join(RAW_DATA_DIR, 'HPb_direct'),
                  short_name='HPb'),
    'HPc': Animal(directory=join(RAW_DATA_DIR, 'HPc_direct'),
                  short_name='HPc'),
    'bon': Animal(directory=join(RAW_DATA_DIR, 'Bond'),
                  short_name='bon'),
    'fra': Animal(directory=join(RAW_DATA_DIR, 'frank'),
                  short_name='fra'),
    'Cor': Animal(directory=join(RAW_DATA_DIR, 'CorrianderData'),
                  short_name='Cor'),
}

# Colors for plots
hls = sns.color_palette('hls', 6)
set1 = sns.color_palette('Set2', 4)

COLORS = {
    'Forward': hls[0],
    'Reverse': hls[1],
    'Inbound': hls[2],
    'Outbound': hls[3],
    'Towards': hls[4],
    'Away': hls[5],
    'Unclassified': 'lightgrey',
    'Inbound-Forward': set1[0],
    'Inbound-Reverse': set1[1],
    'Outbound-Forward': set1[2],
    'Outbound-Reverse': set1[3],
}

# Occupancy normalized histogram parameters
EXTENT = (0, 300, 0, 300)
GRIDSIZE = (60, 60)
