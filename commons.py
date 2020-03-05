DATA_PATH = 'datasets/ecg'
CSV_PATH = 'datasets/ecg/REFERENCE.csv'

sampling_freq = 300

class_ids = {
    'N': 0,
    'O': 1,
    'A': 2,
    '~': 3
}


def get_class_name(class_id):
    for key, value in class_ids.items():
        if value == class_id:
            return key
