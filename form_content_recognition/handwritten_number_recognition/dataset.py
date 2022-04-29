import os
import argparse
import random
import shutil
from shutil import copyfile

#shuffle the mnist and ours
def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s' % dir_path)
    os.makedirs(dir_path)
    print('Create path - %s' % dir_path)


def main(config):

    rm_mkdir(config.train_path)
    rm_mkdir(config.test_path)

    #ours
    filepaths = os.listdir(config.origin_our_path)
    our_list = []
    our_cls = []
    for filepath in filepaths:
        filepath = config.origin_our_path + filepath
        filepath_next = os.listdir(filepath)
        for filename in filepath_next:
            our_list.append(filepath + '/' + filename)

            cls = filepath.split('/')[-1]
            if cls == 'X':
                cls = '10'
            our_cls.append(cls + '_' + filename)

    #mnist
    filepaths = os.listdir(config.origin_mnist_path)
    mnist_list = []
    mnist_cls = []
    d = 1
    for filepath in filepaths:
        filepath = config.origin_mnist_path + filepath
        filepath_next = os.listdir(filepath)
        for filename in filepath_next:
            mnist_list.append(filepath + '/' + filename)
            mnist_cls.append(filename[:4] + str(d) + '.png')
            d += 1

    our_list.extend(mnist_list)
    our_cls.extend(mnist_cls)
    img_list = our_list
    img_cls = our_cls

    num_total = len(img_list)
    num_train = int(config.train_ratio * num_total)
    num_test = num_total - num_train

    print('\nNum of train set : ', num_train)
    print('\nNum of test set : ', num_test)

    Arange = list(range(num_total))
    random.shuffle(Arange)

    for i in range(num_train):

        idx = Arange.pop()
        dst = os.path.join(config.train_path, img_cls[idx])
        copyfile(img_list[idx], dst)

    for i in range(num_test):

        idx = Arange.pop()
        dst = os.path.join(config.test_path, img_cls[idx])
        copyfile(img_list[idx], dst)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.85)
    parser.add_argument('--test_ratio', type=float, default=0.15)

    # data path
    parser.add_argument('--origin_formNumber_path', type=str, default='./form_number_dataset')
    parser.add_argument('--origin_mnist_path', type=str, default='./mnist_dataset')

    parser.add_argument('--train_path', type=str, default='./data/train_img/')
    parser.add_argument('--test_path', type=str, default='./data/test_img/')

    config = parser.parse_args()
    print(config)
    main(config)