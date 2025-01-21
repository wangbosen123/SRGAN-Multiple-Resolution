import os
import tensorflow as tf
import numpy as np
import time
import cv2
from options.base_option import BaseOptions
from preprocess_data import load_path, get_batch_data
from model import build_model, loss
from model.build_model import generator
from utils.mkdir import mkdir
from utils.visualizer import  train_evaluate
from tqdm import tqdm


options = BaseOptions()
options = options.initialize()

class SRGAN():
    def __init__(self):
        mkdir()
        self.options = options
        ## model parameters.
        self.epochs = self.options.epoch
        self.batch_size = self.options.batch
        self.size = self.options.size

        ### get model.
        self.generator = build_model.generator()
        self.discriminator = build_model.discriminator()

        ## get High resolution and Low resolution.
        self.train_ground_truth_path, self.train_L_path = load_path(train=True)
        self.test_ground_truth_path, self.test_L_path = load_path(train=False)
        print(f'Number of Train H : {self.train_ground_truth_path.shape}')
        print(f'Number of Train L : {self.train_L_path.shape}')
        print(f'Number of Test H : {self.test_ground_truth_path.shape}')
        print(f'Number of Test L : {self.test_L_path.shape}')


    def train_step(self, batch_ground_truth, batch_inputs, g_optimizer, d_optimizer):
        with tf.GradientTape(persistent=True) as tape:
            fake_hr = self.generator(batch_inputs, training=True)

            real_validity = self.discriminator(batch_ground_truth, training=True)
            fake_validity = self.discriminator(fake_hr, training=True)

            real_labels = tf.ones_like(real_validity)
            fake_labels = tf.zeros_like(fake_validity)

            d_loss = loss.discriminator_loss(real_labels, fake_labels, real_validity, fake_validity)

            g_loss, c_loss, r_loss = loss.generator_loss(batch_ground_truth, fake_hr, fake_validity)
            total_g_loss = g_loss + c_loss + r_loss


        grads_d = tape.gradient(d_loss, self.discriminator.trainable_variables)
        grads_g = tape.gradient(total_g_loss, self.generator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads_d, self.discriminator.trainable_variables))
        g_optimizer.apply_gradients(zip(grads_g, self.generator.trainable_variables))

        return d_loss, g_loss, c_loss, r_loss


    def train(self):
        d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.options.learning_rate, beta_1=self.options.beta)
        g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.options.learning_rate, beta_1=self.options.beta)


        batch_num = int(self.train_ground_truth_path.shape[0]/self.batch_size)

        d_loss_epoch, g_loss_epoch, c_loss_epoch, r_loss_epoch = [], [], [], []
        for epoch in range(1, self.epochs+1):
            start = time.time()
            d_loss_batch, g_loss_batch, c_loss_batch, r_loss_batch = [], [], [], []
            for batch in tqdm(range(batch_num), desc=f'Epoch {epoch}/{self.epochs}', unit='batch'):
                batch_ground_truth = get_batch_data(self.train_ground_truth_path, batch, self.batch_size)
                batch_inputs = get_batch_data(self.train_L_path, batch, self.batch_size)
                d_loss, g_loss, c_loss, r_loss = self.train_step(batch_ground_truth, batch_inputs, g_optimizer, d_optimizer)
                d_loss_batch.append(d_loss)
                g_loss_batch.append(g_loss)
                c_loss_batch.append(c_loss)
                r_loss_batch.append(r_loss)

            d_loss_epoch.append(np.mean(d_loss_batch))
            g_loss_epoch.append(np.mean(g_loss_batch))
            c_loss_epoch.append(np.mean(c_loss_batch))
            r_loss_epoch.append(np.mean(r_loss_batch))

            print(f' the epoch is {epoch}')
            print(f' the d loss is {d_loss_epoch[-1]}')
            print(f' the g loss is {g_loss_epoch[-1]}')
            print(f' the c loss is {c_loss_epoch[-1]}')
            print(f' the r loss is {r_loss_epoch[-1]}')
            print(f'the spend time is {time.time() - start} second')
            print('----------------------------------')

            frequency_save = self.options.frequency_save
            if epoch % frequency_save == 0 or epoch == 1:
                train_evaluate(epoch, self.generator)
                restore_path = 'result/train' + str(max([int(filename.split('train')[-1]) for filename in os.listdir('result/') if 'train' in filename])) + '/weights/'
                self.generator.save_weights(f'{restore_path}generator-{epoch}.weights.h5')
                self.discriminator.save_weights(f'{restore_path}discriminator-{epoch}.weights.h5')


def test_evaluate():
    g = generator()
    weight_path = options.model_weight
    g.load_weights(weight_path)

    target_dir = (os.getcwd() + '/result/predict' +
                  str(max([int(filename.split('predict')[-1]) for filename in
                           os.listdir(os.getcwd() + '/result') if 'predict' in filename])) + '/')

    if not os.path.exists(target_dir + 'GT'):
        os.mkdir(target_dir + 'GT')
    if not os.path.exists(target_dir + 'LR'):
        os.mkdir(target_dir + 'LR')

    test_HR_path = os.getcwd() + '/' + options.dataroot + 'test_HR/'
    test_LR_path = os.getcwd() + '/' + options.dataroot + 'test_LR/'

    for filename in os.listdir(test_LR_path):
        image = cv2.imread(test_LR_path + filename)
        cv2.imwrite(target_dir + 'GT/' + f'{filename[0:-4]}.jpg', image)

    for filename in os.listdir(test_LR_path):
        image = cv2.imread(test_LR_path + filename) / 255
        image = image.reshape(1, image.shape[0], image.shape[1], 3)
        syn = g(image)
        cv2.imwrite(target_dir + 'LR/' + f'{filename[0:-4]}.jpg', tf.reshape(syn, [syn.shape[1], syn.shape[2], 3]).numpy() * 255)


if __name__ == '__main__':
    import gc
    gc.collect()

    options = BaseOptions()
    options = options.initialize()

    os.environ['CUDA_DEVICES_ORDER'] = 'PIC_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = options.GPU_id
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    Project = SRGAN()

    if options.draw: os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/PycharmProjects/pythonProject/.venv/lib/python3.10/site-packages/PyQt5/Qt5/plugins/platforms'
    print(options.mode)
    if options.mode == 'train':
        Project.train()
    else:
        test_evaluate()


    ## delete train, predict folder in result
    # path = 'result/'
    # import shutil
    # for folder in os.listdir(path):
    #     if 'predict' in folder:
    #         shutil.rmtree(path + folder)

    print('new branch')
























