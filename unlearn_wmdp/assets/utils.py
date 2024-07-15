import pickle

def read_configuration():
    with open('task_config.pkl', 'rb') as f:
        task_config = pickle.load(f)
    return task_config