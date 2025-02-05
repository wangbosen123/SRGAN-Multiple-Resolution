from model.build_model import generator
from options.base_option import BaseOptions
import tritonclient.http as httpclient
import numpy as np
import tensorflow as tf
import time
import requests
import os
import cv2
import matplotlib.pyplot as plt
import concurrent.futures
import gc
import tritonclient.grpc as grpcclient
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/PycharmProjects/pythonProject/.venv/lib/python3.10/site-packages/PyQt5/Qt5/plugins/platforms'


options = BaseOptions()
options = options.initialize()


def Triton_model():
    weight_path = options.model_weight
    G = generator()
    G.load_weights(weight_path)
    save_dir = 'models/SRGAN_model/1/model.savedmodel'
    G.export(save_dir)


def get_config_information():
    model = tf.saved_model.load('models/SRGAN_model/1/model.savedmodel')
    print(model.signatures)


def Dirct_inference():
    os.environ['CUDA_DEVICES_ORDER'] = 'PIC_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = options.GPU_id
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    test_lr_dir = 'datasets/PCB_Background/test_LR'

    images_data = []
    for num, img_name in enumerate(os.listdir(test_lr_dir)):
        if num == 8: break
        img_path = os.path.join(test_lr_dir, img_name)
        img_array = cv2.imread(img_path)
        img_array = img_array / 255.0
        images_data.append(img_array)
    images_data = np.array(images_data)

    g = generator()
    weight_path = options.model_weight
    g.load_weights(weight_path)
    start_time = time.time()

    images_data = images_data.reshape(-1, images_data.shape[1], images_data.shape[2], 3)
    for images in images_data:
        print(images.shape)
        syn = g(images.reshape(-1, images.shape[0], images.shape[1], 3))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(end_time, start_time, len(images_data))
    print(f"Direct Server 推論速度： {end_time - start_time} second")


def Triton_inference():
    triton_url = 'http://localhost:8000/v2/models/SRGAN_model/infer'
    test_lr_dir = 'datasets/PCB_Background/test_LR'

    images_data = []
    for num, img_name in enumerate(os.listdir(test_lr_dir)):
        if num == 8: break
        img_path = os.path.join(test_lr_dir, img_name)
        img_array = cv2.imread(img_path)
        img_array = img_array / 255.0
        images_data.append(img_array)
    print(np.array(images_data).shape)

    data = {
        "inputs": [
            {
                "name": "keras_tensor",  # 根據模型設置修改 input 名稱
                "shape": [len(images_data), *images_data[0].shape],  # 這裡會自動處理批次大小
                "datatype": "FP32",  # 根據模型的數據類型來設置
                "data": [img.tolist() for img in images_data]
            }
        ]
    }
    print(data["inputs"][0]["shape"])
    start_time = time.time()
    response = requests.post(triton_url, json=data)
    if response.status_code != 200:
        print(f"推論錯誤: {response.text}")

    # response_data = response.json()
    # output_shape = response_data['outputs'][0]['shape']
    # output_images = np.array(output_data).reshape(output_shape)
    # print(output_images.shape)

    end_time = time.time()
    print(f"Triton Server 推論速度： {end_time - start_time} second")


class test_triton():
    def __init__(self):
        self.input_data = tf.keras.applications.vgg16.preprocess_input(np.random.rand(1000, 64, 64, 3).astype(np.float32))
        self.vgg_model = tf.keras.applications.VGG16(weights='imagenet', input_shape=(64, 64, 3), include_top=False)
        self.vgg_model.export("models/VGG_model/1/model.savedmodel/")


    def get_triton_model_name(self):
        model = tf.saved_model.load('models/VGG_model/1/model.savedmodel')
        print(model.signatures)

    def Direct_inference(self):
        batch_size, batch = 16, 0
        batch_data = []
        initial_time = time.time()
        for data in self.input_data:
            batch_data.append(data)
            if len(batch_data) == batch_size:
                batch += 1
                start_time = time.time()
                data = np.array(batch_data).reshape(-1, self.input_data.shape[1], self.input_data.shape[2], self.input_data.shape[3])
                _ = self.vgg_model(data)
                print(f'{batch} successful!')
                print(f'{batch} inference time : {time.time() - start_time} second')
                batch_data = []

        if batch_data:
            start_time = time.time()
            data = np.array(batch_data).reshape(-1, self.input_data.shape[1], self.input_data.shape[2],
                                                self.input_data.shape[3])
            _ = self.vgg_model(data)
            print(f'{batch} successful!')
            print(f'{batch} inference time : {time.time() - start_time} second')


        end = time.time()
        print(f'Direct inference time: {end - initial_time} second')

    def Triton_inference(self):
        triton_url = 'http://localhost:8000/v2/models/VGG_model/infer'
        batch_size, batch = 16, 0
        batch_data = []
        initial_time = time.time()
        for num, data in enumerate(self.input_data):
            batch_data.append(data)
            if len(batch_data) == batch_size:
                batch += 1
                batch_data = np.array(batch_data).reshape(-1, self.input_data.shape[1], self.input_data.shape[2], self.input_data.shape[3])
                data = {
                    "inputs": [
                        {
                            "name": "keras_tensor",
                            "shape": [len(batch_data), self.input_data.shape[1], self.input_data.shape[2], self.input_data.shape[3]],
                            "datatype": "FP32",  # 根據模型的數據類型來設置
                            "data": [img.tolist() for img in batch_data]
                        }
                    ]
                }
                start_time = time.time()
                response = requests.post(triton_url, json=data)
                if response.status_code != 200:
                    print(f"推論錯誤: {response.text}")
                else:
                    print(f'{batch} successful!')
                    print(f'{batch} inference time : {time.time() - start_time} second')
                batch_data = []
                del response
                gc.collect()

        if batch_data:
            batch_data = np.array(batch_data).reshape(-1, self.input_data.shape[1], self.input_data.shape[2],
                                                      self.input_data.shape[3])
            data = {
                "inputs": [
                    {
                        "name": "keras_tensor",
                        "shape": [len(batch_data), self.input_data.shape[1], self.input_data.shape[2], self.input_data.shape[3]],
                        # 這裡會自動處理批次大小
                        "datatype": "FP32",
                        "data": [img.tolist() for img in batch_data]
                    }
                ]
            }
            start_time = time.time()
            response = requests.post(triton_url, json=data)
            if response.status_code != 200:
                print(f"推論錯誤: {response.text}")
            else:
                print(f'{batch} successful!')
                print(f'{batch} inference time : {time.time() - start_time} second')

        end_time = time.time()
        print(f'Triton Server time: {end_time - initial_time} second')


    def triton_inference_batch(self, batch_data):
        triton_url = 'http://localhost:8000/v2/models/VGG_model/infer'
        data = {
            "inputs": [
                {
                    "name": "keras_tensor",
                    "shape": [len(batch_data), 64, 64, 3],
                    "datatype": "FP32",
                    "data": [img.tolist() for img in batch_data]
                }
            ]
        }
        response = requests.post(triton_url, json=data)
        return response


    def run_inference(self):
        batch_size = 16
        batch = 0
        batch_data = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_batch = {}
            for num, data in enumerate(self.input_data):
                batch_data.append(data)
                if len(batch_data) == batch_size:
                    future = executor.submit(self.triton_inference_batch, batch_data)
                    future_to_batch[future] = batch_data
                    batch_data = []

            initial_time = time.time()
            for future in concurrent.futures.as_completed(future_to_batch):
                start = time.time()
                response = future.result()
                if response.status_code != 200:
                    print(f"推論錯誤: {response.text}")
                else:
                    print(f"Batch {batch} successful!")
                    print(f"Inference time: {time.time() - start} seconds")

        end_time = time.time()
        print(f'TritonServer inference time : {end_time - initial_time} second.')




if __name__ == '__main__':
    Project = test_triton()
    # Project.get_triton_model_name()
    Project.Direct_inference()
    Project.Triton_inference()
    # tf.keras.backend.clear_session()
    # gc.collect()
    # Dirct_inference()
    # tf.keras.backend.clear_session()
    # gc.collect()
    # Triton_inference()






