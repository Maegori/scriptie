import os

# Define the musical styles
genre = [
    '1956',
    '1957',
    '1958',
    '1959',
    '1960',
    '1961',
    '1962',
    '1963',
    '1964',
    '1965',
    '1966',
    '1967',
    '1968',
    '1969',
    '1970',
    '1971',
    '1972',
    '1973',
    '1974',
    '1975',
    '1976',
    '1977',
    '1978',
    '1979',
    '1980',
    '1981',
    '1982',
    '1983',
    '1984',
    '1985',
    '1986',
    '1987',
    '1988',
    '1989',
    '1990',
    '1991',
    '1992',
    '1993',
    '1994',
    '1995',
    '1996',
    '1997',
    '1998',
    '1999',
    '2000',
    '2001',
    '2002',
    '2003',
    '2004',
    '2005',
    '2006',
    '2007',
    '2008',
    '2009',
    '2010',
    '2011',
    '2012',
    '2013',
    '2014',
    '2015',
    '2016',
    '2017',
    '2018',
    '2019',
    '2020'
]

styles = [
    ['../../midi_output/1956'],
    ['../../midi_output/1957'],
    ['../../midi_output/1958'],
    ['../../midi_output/1959'],
    ['../../midi_output/1960'],
    ['../../midi_output/1961'],
    ['../../midi_output/1962'],
    ['../../midi_output/1963'],
    ['../../midi_output/1964'],
    ['../../midi_output/1965'],
    ['../../midi_output/1966'],
    ['../../midi_output/1967'],
    ['../../midi_output/1968'],
    ['../../midi_output/1969'],
    ['../../midi_output/1970'],
    ['../../midi_output/1971'],
    ['../../midi_output/1972'],
    ['../../midi_output/1973'],
    ['../../midi_output/1974'],
    ['../../midi_output/1975'],
    ['../../midi_output/1976'],
    ['../../midi_output/1977'],
    ['../../midi_output/1978'],
    ['../../midi_output/1979'],
    ['../../midi_output/1980'],
    ['../../midi_output/1981'],
    ['../../midi_output/1982'],
    ['../../midi_output/1983'],
    ['../../midi_output/1984'],
    ['../../midi_output/1985'],
    ['../../midi_output/1986'],
    ['../../midi_output/1987'],
    ['../../midi_output/1988'],
    ['../../midi_output/1989'],
    ['../../midi_output/1990'],
    ['../../midi_output/1991'],
    ['../../midi_output/1992'],
    ['../../midi_output/1993'],
    ['../../midi_output/1994'],
    ['../../midi_output/1995'],
    ['../../midi_output/1996'],
    ['../../midi_output/1997'],
    ['../../midi_output/1998'],
    ['../../midi_output/1999'],
    ['../../midi_output/2000'],
    ['../../midi_output/2001'],
    ['../../midi_output/2002'],
    ['../../midi_output/2003'],
    ['../../midi_output/2004'],
    ['../../midi_output/2005'],
    ['../../midi_output/2006'],
    ['../../midi_output/2007'],
    ['../../midi_output/2008'],
    ['../../midi_output/2009'],
    ['../../midi_output/2010'],
    ['../../midi_output/2011'],
    ['../../midi_output/2012'],
    ['../../midi_output/2013'],
    ['../../midi_output/2014'],
    ['../../midi_output/2015'],
    ['../../midi_output/2016'],
    ['../../midi_output/2017'],
    ['../../midi_output/2018'],
    ['../../midi_output/2019'],
    ['../../midi_output/2020'],
]

# styles = [
#     [
#         'data/baroque/bach',
#         'data/baroque/handel',
#         'data/baroque/pachelbel'
#     ],
#     [
#         'data/classical/burgmueller',
#         'data/classical/clementi',
#         'data/classical/haydn',
#         'data/classical/beethoven',
#         'data/classical/brahms',
#         'data/classical/mozart'
#     ],
#     [
#         'data/romantic/balakirew',
#         'data/romantic/borodin',
#         'data/romantic/brahms',
#         'data/romantic/chopin',
#         'data/romantic/debussy',
#         'data/romantic/liszt',
#         'data/romantic/mendelssohn',
#         'data/romantic/moszkowski',
#         'data/romantic/mussorgsky',
#         'data/romantic/rachmaninov',
#         'data/romantic/schubert',
#         'data/romantic/schumann',
#         'data/romantic/tchaikovsky',
#         'data/romantic/tschai'
#     ]
# ]

NUM_STYLES = sum(len(s) for s in styles)

# MIDI Resolution
DEFAULT_RES = 96
MIDI_MAX_NOTES = 128
MAX_VELOCITY = 127

# Number of octaves supported
NUM_OCTAVES = 4
OCTAVE = 12

# Min and max note (in MIDI note number)
MIN_NOTE = 36
MAX_NOTE = MIN_NOTE + NUM_OCTAVES * OCTAVE
NUM_NOTES = MAX_NOTE - MIN_NOTE

# Number of beats in a bar
BEATS_PER_BAR = 4
# Notes per quarter note
NOTES_PER_BEAT = 4
# The quickest note is a half-note
NOTES_PER_BAR = NOTES_PER_BEAT * BEATS_PER_BAR

# Training parameters
BATCH_SIZE = 16
SEQ_LEN = 8 * NOTES_PER_BAR

# Hyper Parameters
OCTAVE_UNITS = 64
STYLE_UNITS = 64
NOTE_UNITS = 3
TIME_AXIS_UNITS = 256
NOTE_AXIS_UNITS = 128

TIME_AXIS_LAYERS = 2
NOTE_AXIS_LAYERS = 2

# Move file save location
OUT_DIR = 'out'
MODEL_DIR = os.path.join(OUT_DIR, 'models')
MODEL_FILE = os.path.join(OUT_DIR, 'model.h5')
SAMPLES_DIR = os.path.join(OUT_DIR, 'samples')
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
