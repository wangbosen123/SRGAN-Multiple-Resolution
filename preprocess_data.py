import numpy as np
import os
import cv2
from options import base_option
from options.base_option import BaseOptions


def load_path(train=True):
    options = BaseOptions()
    opt = options.initialize()


    path = opt.dataroot
    H_path, L_path = [], []

    if train: name = 'train'
    else: name = 'test'

    for folder in os.listdir(path):
        if name in folder and 'LR' in folder:
            for filename in os.listdir(path + folder):
                    L_path.append(path+folder+'/'+filename)
                    H_path.append(path+folder.split('_')[0]+'_HR/'+filename.split('_')[0]+'_'+filename.split('_')[1]+'.jpg')

    overall_data = list(zip(H_path, L_path))
    np.random.shuffle(overall_data)
    overall_data = list(zip(*overall_data))
    return np.array(overall_data[0]), np.array(overall_data[1])



def get_batch_data(data, batch_idx, batch):
    range_min = batch * batch_idx
    range_max = len(data) if batch * (batch_idx + 1) > len(data) else batch * (batch_idx + 1)

    batch_data = []
    for idx in range(range_min, range_max):
        image = cv2.imread(data[idx]) /255
        batch_data.append(image)

    batch_data = np.array(batch_data)
    return batch_data.reshape(-1, batch_data.shape[1], batch_data.shape[2], 3)




if __name__ == '__main__':
    train_H_path, train_L_path = load_path(train=True)
    print(train_H_path.shape, train_L_path.shape)
    print(train_H_path)
    print(train_L_path)

    for batch_idx in range(0, 28):
        print(batch_idx)
        image = get_batch_data(train_H_path, 130, batch_idx)
        print(image.shape)

    test_H_path, test_L_path = load_path(train=False)
    print(test_H_path.shape, test_L_path.shape)