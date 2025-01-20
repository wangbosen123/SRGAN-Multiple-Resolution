import matplotlib.pyplot as plt
import os
import cv2
from options.base_option import BaseOptions
import tensorflow as tf

options = BaseOptions()
options = options.initialize()

def loss_curve(d_loss, g_loss, c_loss, r_loss):
    previous_dir = os.path.dirname(os.getcwd())
    restore_dir = f'{previous_dir}/result/train' + str(max([int(filename.split('train')[-1]) for filename in os.listdir(previous_dir + '/result') if 'train' in filename])) + '/'

    plt.plot(d_loss)
    plt.title('d loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss value')
    plt.savefig(f'{restore_dir}d loss')
    plt.close()

    plt.plot(g_loss)
    plt.title('g loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss value')
    plt.savefig(f'{restore_dir}g loss')
    plt.close()

    plt.plot(c_loss)
    plt.title('content loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss value')
    plt.legend(['d_loss', 'g_loss'], loc='upper right')
    plt.savefig(f'{restore_dir}content loss')
    plt.close()

    plt.plot(r_loss)
    plt.title('image loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss value')
    plt.savefig(f'{restore_dir}image loss')
    plt.close()



def train_evaluate(epoch, generator):
    path = options.dataroot
    if __name__ == '__main__':
        test_HR_path = os.path.dirname(os.getcwd()) + '/' + path + 'test_HR/'
    else:
        test_HR_path = os.getcwd() + '/' + path + 'test_HR/'
    number_resolution_type = options.resolution_type_number
    resolution_type = []
    number_resolution_type = number_resolution_type.split(',')
    for re in number_resolution_type:
        resolution_type.append(re)

    import random
    select_index = [random.randint(0, len(test_HR_path)-1) for _ in range(1)]

    select_HR_path = [test_HR_path + os.listdir(test_HR_path)[idx] for idx in select_index]
    select_LR_path = [path.split('.jpg')[0].replace('HR', 'LR') + f'_{resolution_type}.jpg' for path in select_HR_path for resolution_type in number_resolution_type]
    print(select_HR_path)
    print(select_LR_path)

    plt.subplots(figsize=(50, 20))
    plt.axis('off')
    plt.subplots_adjust(hspace=0, wspace=0)
    real_image = cv2.imread(select_HR_path[0])
    plt.subplot(2, len(select_LR_path) + 1, 1)
    plt.imshow(real_image)
    plt.axis('off')

    for num, path in enumerate(select_LR_path):
        image = cv2.imread(path) /255
        plt.subplot(2, len(select_LR_path) + 1, num+2)
        plt.axis('off')
        plt.imshow(image)

        syn = generator(image.reshape(1, image.shape[0], image.shape[1], 3))
        plt.subplot(2, len(select_LR_path) + 1, len(select_LR_path)+3+num)
        plt.axis('off')
        plt.imshow(tf.reshape(syn, [syn.shape[1], syn.shape[2], 3]))

    if __name__ == '__main__':
        plt.savefig(os.path.dirname(os.getcwd())+'/result'+ '/train' +str(max([int(filename.split('train')[-1]) for filename in os.listdir(os.path.dirname(os.getcwd())+'/result')
                                                                            if 'train' in filename])) + '/' + f'EPOCH{epoch}_RESULT')
    else:
        plt.savefig(os.getcwd() + '/result' + '/train' + str(max([int(filename.split('train')[-1]) for filename in os.listdir(os.getcwd()+'/result')
                                                                  if 'train' in filename])) + '/' + f'EPOCH{epoch}_RESULT')
    plt.close()



if __name__ == '__main__':
    epoch, generator = 0, 0
    train_evaluate(epoch, generator)