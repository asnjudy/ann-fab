import os
import argparse
import lmdb
import numpy as np
import matplotlib.pyplot as plt

import proto_defs_pb2 as proto


def is_1d(x):
    if np.isscalar(x):
        return False

    nd = len(x.shape)
    if nd == 1:
        return True
    elif np.all(np.array(x.shape[1:]) == 1):
        return True
    return False


def parse_command_line(use_parser=None):
    # setup the parser
    if use_parser is None:
        parser = argparse.ArgumentParser(
            description="Show images in an lmdb database")
    else:
        parser = use_parser

    parser.add_argument("--data", required=True,
                        help="The root of the database to read")
    parser.add_argument('--all', dest='all', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument("--key", required=False,
                        help="The key you want to view")
    return parser.parse_args()


def is_lmdb_folder(folder):

    if not os.path.isdir(folder):
        return False

    required_files = ['data.mdb', 'lock.mdb']
    for f in required_files:
        if not os.path.isfile(os.path.join(folder, f)):
            return False

    return True


def unitvec(vec):
    vec = np.asarray(vec, dtype=float)
    veclen = np.linalg.norm(vec)
    if veclen > 0.0:
        return vec / veclen
    else:
        return vec


def value_to_image(value):
    datum = proto.Datum()
    datum.ParseFromString(value)
    return np.asarray(datum_to_image(datum)).ravel()


def normalized_image_vector(value):
    return unitvec(value_to_image(value))


def plot_image_from_datum(key, datum):
    a = datum_to_image(datum)
    plt.imshow(a)
    plt.title(key)


def open_database(database):
    if not is_lmdb_folder(database):
        raise Exception("The data path (%s) is not a valid LMDB database" %
                        database)

    return lmdb.open(database)


def datum_to_image(datum):
    x = np.fromstring(datum.data,
                      dtype=np.uint8).reshape(datum.channels, datum.height,
                                              datum.width)

    a = np.zeros((datum.height, datum.width, datum.channels), dtype=np.float32)
    for c in range(datum.channels):
        for h in range(datum.height):
            a[h, :, c] = x[c, h, :] / 255.

    return a


def image_to_datum(image):
    datum = proto.Datum()
    if len(image.shape) == 3:
        datum.channels = image.shape[0]
        datum.height = image.shape[1]
        datum.width = image.shape[2]
    else:
        datum.channels = 1
        datum.height = image.shape[0]
        datum.width = image.shape[1]

    datum.data = image.tostring()

    return datum
