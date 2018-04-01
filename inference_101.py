"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

from PIL import Image

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel_101, ImageReader, decode_labels, prepare_label
from deeplab_resnet import utils

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

snapshot_dir = './DeepLab2/'
Test_Type = "Origin"
NUM_CLASSES = 2
test_batch_num = 85
if Test_Type == "Origin":
    pic_width = 648
    pic_height = 486
    SAVE_DIR = snapshot_dir + 'output/Origin/'
    test_set = './Data/Practical_Origin_Eye_image/'
if Test_Type == "Non_Origin":
    pic_width = 800
    pic_height = 800
    SAVE_DIR = snapshot_dir + 'output/Non_Origin/'
    test_set = './Data/Practical_Eye_image/'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("--test_set", type=str, default=test_set,
                        help="Choose test set.")
    parser.add_argument("--model_weights", type=str, default=snapshot_dir,
                        help="Path to the file with model weights.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    parser.add_argument("--GPU", type=str, default='3',
                        help="The number of GPU you want to use.")
    return parser.parse_args()


def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    # saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    img_list = utils.generate_filelist_test(args.test_set)
    img_queue = tf.train.slice_input_producer([img_list], shuffle=False)
    img_content = tf.read_file(img_queue[0])

    # Prepare image.
    img = tf.image.decode_png(img_content, channels=3)
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN
    img = tf.reshape(img, [pic_height, pic_width, 3])

    image_test, image_name = tf.train.batch([img, img_queue[0]], 1)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Create network.
    net = DeepLabResNetModel_101({'data': image_test}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2, ])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Set up TF session and initialize variables.

    init = tf.global_variables_initializer()

    sess.run(init)

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.model_weights)

    for time in range(test_batch_num):
        # Perform inference.

        preds, test_img_name = sess.run([pred, image_name])

        test_img_name = test_img_name[0].decode()
        test_img_name = test_img_name.split('/')[-1].strip('.png\n')
        print(test_img_name)

        msk = decode_labels(preds, num_classes=args.num_classes)

        im = Image.fromarray(msk[0])
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        im.save(args.save_dir + test_img_name + '.png')

        print('The output file has been saved to {}'.format(args.save_dir + 'mask.png'))
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
