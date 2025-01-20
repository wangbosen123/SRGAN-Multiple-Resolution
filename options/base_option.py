import argparse


class BaseOptions():
    def __init__(self):
        self.initialized = False


    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--dataroot', type=str, default='datasets/PCB_Background/', help='path to image')
        parser.add_argument('--epoch', type=int, default=3, help='training times')
        parser.add_argument('--batch', type=int, default=1,)
        parser.add_argument('--size', default=640, help='image size')
        parser.add_argument('--GPU-id', type=str, default='0,1', help='which GPU do you want to use?')
        parser.add_argument('--resolution-type-number', default='X2,X4,X64', help='how many resolution do you synthesis.')
        # parser.add_argument('--train_number', type=int, default=500)

        parser.add_argument('--learning-rate', type=float, default=1e-4)
        parser.add_argument('--beta', type=float, default=0.5)
        parser.add_argument('--frequency-save', type=int, default=10, help='What frequency do you want to save model?')

        parser.add_argument('--mode', type=str, default='train', help='train or predict')
        parser.add_argument('--draw', type=bool, default=True, help='Normal set the False')
        parser.add_argument('--train-weight', type=bool, default='result/train1/weights/generator-weights-200.h5')
        parser.add_argument('--model-weight', type=str, default='result/train1/weights/generator.weights-200.h5')

        options = parser.parse_args()
        return options


if __name__ == '__main__':
    options = BaseOptions()
    options = options.initialize()
    print(options)
    print(options.dataroot)

