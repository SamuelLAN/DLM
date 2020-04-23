import os
import shutil
from lib import utils

"""
Cleaning the useless model files and tensorboard files
"""

save_best_model_num = 1

model_dir = utils.create_dir_in_root('runtime', 'models')
tb_dir = utils.create_dir_in_root('runtime', 'tensorboard')
tokenizer_dir = utils.create_dir_in_root('runtime', 'tokenizer')

for model_name in os.listdir(model_dir):
    tmp_model_dir = os.path.join(model_dir, model_name)
    print(f'\nchecking {tmp_model_dir} ...')

    for _date in os.listdir(tmp_model_dir):
        date_dir = os.path.join(tmp_model_dir, _date)
        model_list = os.listdir(date_dir)

        print(f'\tchecking {_date}')

        # if model dir is empty, delete the model dir and its tensorboard files
        if not model_list:
            tmp_tb_dir = os.path.join(tb_dir, model_name, _date)
            if os.path.exists(tmp_tb_dir):
                shutil.rmtree(tmp_tb_dir)
            tmp_tok_dir = os.path.join(tokenizer_dir, model_name, _date)
            if os.path.exists(tmp_tok_dir):
                shutil.rmtree(tmp_tok_dir)
            os.removedirs(date_dir)
            print(f'\tremove {_date}')

        # if model dir is not empty, only save the best models
        else:
            model_list.sort(reverse=True)
            for model_file_name in model_list[1:]:
                model_path = os.path.join(date_dir, model_file_name)
                os.remove(model_path)
                print(f'\t\tremove {model_file_name}')

for model_name in os.listdir(tb_dir):
    tmp_tb_dir = os.path.join(tb_dir, model_name)
    print(f'\nchecking {tmp_tb_dir} ...')

    for _date in os.listdir(tmp_tb_dir):
        date_dir = os.path.join(tmp_tb_dir, _date)
        tmp_model_dir = os.path.join(model_dir, model_name, _date)

        print(f'\tchecking {_date}')

        if not os.path.exists(tmp_model_dir):
            shutil.rmtree(date_dir)
            print(f'\tremove {_date}')

for model_name in os.listdir(tokenizer_dir):
    tmp_tok_dir = os.path.join(tokenizer_dir, model_name)
    print(f'\nchecking {tmp_tok_dir} ...')

    for _date in os.listdir(tmp_tok_dir):
        date_dir = os.path.join(tmp_tok_dir, _date)
        tmp_model_dir = os.path.join(model_dir, model_name, _date)

        print(f'\tchecking {_date}')

        if not os.path.exists(tmp_model_dir):
            shutil.rmtree(date_dir)
            print(f'\tremove {_date}')

print('\ndone')
