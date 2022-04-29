import os
import cv2
import numpy as np

TRAIN_IMAGES_DIR = "./data/mnist/train/"
TEST_IMAGES_DIR = "./data/mnist/test/"

TRAIN_IMAGE_DATASET_PATH = "./data/mnist/MNIST/raw/train-images-idx3-ubyte"
TRAIN_LABEL_DATASET_PATH = "./data/mnist/MNIST/raw/train-labels-idx1-ubyte"
TEST_IMAGE_DATASET_PATH = "./data/mnist/MNIST/raw/t10k-images-idx3-ubyte"
TEST_LABEL_DATASET_PATH = "./data/mnist/MNIST/raw/t10k-labels-idx1-ubyte"


def convert_to_image(dataset_type):
    if dataset_type == "train":
        images_dir = TRAIN_IMAGES_DIR
        image_dataset = open(TRAIN_IMAGE_DATASET_PATH, "rb")
        label_dataset = open(TRAIN_LABEL_DATASET_PATH, "rb")
    elif dataset_type == "test":
        images_dir = TEST_IMAGES_DIR
        image_dataset = open(TEST_IMAGE_DATASET_PATH, "rb")
        label_dataset = open(TEST_LABEL_DATASET_PATH, "rb")
    else:
        print("Invalid type.")
        return

    counter = [0] * 10

    image_magic_number = int.from_bytes(image_dataset.read(4), byteorder='big', signed=False)
    image_num = int.from_bytes(image_dataset.read(4), byteorder='big', signed=False)
    image_row = int.from_bytes(image_dataset.read(4), byteorder='big', signed=False)
    image_col = int.from_bytes(image_dataset.read(4), byteorder='big', signed=False)

    # for i in range(10):
    #     if not os.path.exists(images_dir + str(i)):
    #         os.makedirs(images_dir + str(i))

    for i in range(image_num):
        image = []
        for j in range(image_row * image_col):
            image.append(int.from_bytes(image_dataset.read(1), byteorder='big', signed=False))

        image = np.array(image, dtype=np.uint8).reshape((image_row, image_col))
        image = ~image

        label = int.from_bytes(label_dataset.read(1), byteorder='big', signed=False)

        counter[label] += 1

        image_path = images_dir + str(label) + "_" + 'mn' + str(counter[label]) + ".png"

        cv2.imwrite(image_path, image)

        if (i + 1) % 1000 == 0:
            print("Running, " + dataset_type + " images: " + str(i + 1) + "/" + str(image_num))

        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    image_dataset.close()
    label_dataset.close()

    print(dataset_type + " dataset finished.")


if __name__ == "__main__":
    convert_to_image("train")
    convert_to_image("test")
    print("All finished.")