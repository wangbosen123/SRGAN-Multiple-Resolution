import os
from options.base_option import BaseOptions


def mkdir():
    options = BaseOptions()
    options = options.initialize()
    if __name__ == '__main__':
        previous_dir = os.path.dirname(os.getcwd())
        exits_train_dir = os.listdir(previous_dir + '/result')
    else:
        previous_dir = os.getcwd()
        exits_train_dir = os.listdir(previous_dir + '/result')

    if options.mode == 'train':
        create_dir = 'train'
    else:
        create_dir = 'predict'

    path_exits = False
    for name in exits_train_dir:
        if create_dir in name:
            path_exits = True
            break
        else:
            continue

    if not path_exits:
        os.mkdir(previous_dir + f'/result/{create_dir}1')
        if options.mode == 'train':
            os.mkdir(previous_dir + f'/result/{create_dir}1/weights')
    else:
        max_number = max([int(train_num.split(create_dir)[-1]) for train_num in exits_train_dir if create_dir in train_num])
        os.mkdir(f'{previous_dir}/result/{create_dir}{max_number+1}')
        if options.mode == 'train':
            os.mkdir(f'{previous_dir}/result/{create_dir}{max_number + 1}/weights')


if __name__ == '__main__':
    mkdir()


