#!/usr/bin/env python2
# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import sys
import time
import lmdb
import struct
import argparse
import os
import numpy as np

import config

import annfab.utils
from mnist import MnistDownloader


class MnistLMDB(MnistDownloader):

    def processData(self):
        print "Processing data"
        self.__extract_images('train-images.bin', 'train-labels.bin', 'train')
        self.__extract_images('test-images.bin', 'test-labels.bin', 'test')

    def __extract_images(self, images_file, labels_file, phase):
        """
        Extract information from binary files and store them in the LMDB.
        """
        images, max_size = self.__readImages(
            os.path.join(self.outdir, images_file))

        map_size = len(images) * max_size * 10
        env = lmdb.open(self.outdir, map_size=map_size)

        with env.begin(write=True) as txn:
            # txn is a Transaction object
            for i, image in enumerate(images):
                datum = annfab.utils.image_to_datum(image)
                str_id = '{:08}'.format(i)

                # The encode is only essential in Python 3
                txn.put(str_id.encode('ascii'), datum.SerializeToString())

    def __readImages(self, filename):
        """
        Returns a list of numpy arrays
        """
        print 'Reading images from %s ...' % filename
        images = []
        max_size = 0
        with open(filename, 'rb') as infile:
            infile.read(4)  # ignore magic number
            count = struct.unpack('>i', infile.read(4))[0]
            rows = struct.unpack('>i', infile.read(4))[0]
            columns = struct.unpack('>i', infile.read(4))[0]

            for i in xrange(count):
                data = infile.read(rows*columns)
                image = np.fromstring(data, dtype=np.uint8)
                image = image.reshape((rows, columns))
                image = 255 - image  # now black digit on white background
                images.append(image)
                max_size = max(max_size, image.size)
        return images, max_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download dataset')

    # Positional arguments
    # parser.add_argument('dataset',
    #                     help='mnist/cifar10/cifar100'
    #                     )
    # parser.add_argument('output_dir',
    #                     help='The output directory for the data'
    #                     )

    # Optional arguments
    parser.add_argument('-c', '--clean',
                        action='store_true',
                        help='clean out the directory first (if it exists)'
                        )

    args = vars(parser.parse_args())

    dataset = 'mnist'
    output_dir = 'mnist_data'

    start = time.time()
    if dataset == 'mnist':
        d = MnistLMDB(
            outdir=output_dir,
            clean=args['clean'])
        d.getData()
    else:
        print 'Unknown dataset "%s"' % args['dataset']
        sys.exit(1)

    print 'Done after %s seconds.' % (time.time() - start)
