import shutil
from transformers import DataProcessor, InputExample
import os
import random
from pathlib import Path

tasks = ['MNLI', 'RTE', 'QQP', 'subj', 'SST-2', 'mpqa']

for task in tasks:

    if task in ['MNLI', 'RTE', 'QQP', 'SST-2']:
        file_suff = '.tsv'

        if task == 'MNLI':
            pass

    else:
        file_suff = '.csv'

    data_folder = os.path.join(os.getcwd() + '/original', task)
    full_folder = os.path.join(os.getcwd() + '/full', task)
    Path(full_folder).mkdir(parents=True, exist_ok=True)
    for file in os.listdir(data_folder):
        origin_file = os.path.join(data_folder, file)
        dest_file = os.path.join(full_folder, file.replace('dev', 'test'))
        shutil.copy(origin_file, dest_file)

    train_file = os.path.join(data_folder, 'train' + file_suff)
    dest_folder = os.path.join(os.getcwd() + '/clue', task)
    for subfolder in os.listdir(dest_folder):
        dest_path = os.path.join(dest_folder, subfolder)
        dest_path = os.path.join(dest_path, 'un_train' + file_suff)
        #os.remove(dest_path)
        shutil.copy(train_file, dest_path)