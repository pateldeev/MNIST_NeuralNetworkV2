import cv2
import numpy as np
from pathlib import Path


# Returns a list of images of images from a compressed ubyte file (as specified by http://yann.lecun.com/exdb/mnist/)
# Each image is a 2D list of pixel values.
def get_images(file_directory):
    with open(file_directory, 'rb') as input_file:
        magic_number = int.from_bytes(input_file.read(4), byteorder="big")
        num_img = int.from_bytes(input_file.read(4), byteorder="big")
        rows = int.from_bytes(input_file.read(4), byteorder="big")
        cols = int.from_bytes(input_file.read(4), byteorder="big")

        if magic_number != 2051:
            raise ValueError("Unknown Magic Number: {}".format(magic_number))

        return [[[int.from_bytes(input_file.read(1), byteorder="big")
                  for _c in range(cols)]
                 for _r in range(rows)]
                for _i in range(num_img)]


# Returns a list of ground truth labels from a compressed ubyte file (as specified by http://yann.lecun.com/exdb/mnist/)
# Each label will indicate the numerical value of the corresponding image.
def get_labels(file_directory):
    with open(file_directory, 'rb') as input_file:
        magic_number = int.from_bytes(input_file.read(4), byteorder="big")
        num_items = int.from_bytes(input_file.read(4), byteorder="big")

        if magic_number != 2049:
            raise ValueError("Unknown Magic Number: {}".format(magic_number))

        return [int.from_bytes(input_file.read(1), byteorder="big") for _ in range(num_items)]


# Saves a list of images to a directory. Assumes each image is a 2D list of pixel values in [0, 255].
def save_images(images, base_dir, base_name="img_"):
    num_digits = len(str(len(images)))
    save_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    for img_num, img in enumerate(images):
        cv2.imwrite("{}{}{}.png".format(base_dir, base_name, str(img_num).zfill(num_digits)),
                    np.array(img, dtype=np.uint8), save_params)


# Reads a list of images to a directory. Each image is a 2D list of pixel values in [0, 255].
def read_images(base_dir, base_name="img_", read_limit=0):
    p_list = sorted(Path(base_dir).glob('**/{}*.png'.format(base_name)), key=lambda p: str(p))

    if read_limit <= 0:
        return [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).tolist() for p in p_list]
    else:
        return [cv2.imread(str(p_list[i]), cv2.IMREAD_GRAYSCALE).tolist() for i in range(read_limit)]
