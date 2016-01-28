import lmdb
import matplotlib.pyplot as plt
import argparse

import config

import annfab.proto_defs_pb2 as proto
import annfab.utils


def show_image(key, datum):
    print 'key = ', key

    annfab.utils.plot_image_from_datum(key, datum)
    plt.show()


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="Show images in an lmdb database")

    parser.add_argument("--nfile",
                        default=None,
                        help="A file containing a list of neigbours.")

    return parser


def plot_images(env, old_key, keys_to_show, ds):
    if old_key is None:
        return

    if len(keys_to_show) == 0:
        return

    datum = proto.Datum()

    fig = plt.figure()

    keys = [old_key, ]
    keys.extend(keys_to_show)
    d = [0.0, ]
    d.extend(ds)

    sp = 100 + (len(keys)) * 10

    with env.begin() as txn:
        cursor = txn.cursor()
        for i in range(len(keys)):
            print 'key=', keys[i]
            k = cursor.set_key(keys[i])
            assert k, 'key ' + keys[i] + ' does not exist'
            key, value = cursor.item()

            datum.ParseFromString(value)
            plt.subplot(sp + i + 1)
            annfab.utils.plot_image_from_datum(key, datum)
            plt.title("%s : %f" % (key, d[i]))

    plt.show()


def process_neighbours(args):

    env = annfab.utils.open_database(args.data)

    with open(args.nfile, 'r') as f:
        lines = f.readlines()

    old_key = None
    keys_to_show = []
    ds = []
    for l in lines:
        try:
            parts = l.split(',')
            key = parts[0].strip()
            n_key = parts[1].strip()
            d = parts[2].strip()
        except IndexError:
            print 'The line %s could not be split' % l
            continue

        if key != old_key:
            plot_images(env, old_key, keys_to_show, ds)
            old_key = key
            keys_to_show = []
            ds = []

        if key == n_key:
            continue

        d = abs(float(d))
        if d < 0.01:
            print key, n_key, d
            keys_to_show.append(n_key)
            ds.append(d)


def main():

    parser = create_arg_parser()
    args = annfab.utils.parse_command_line(parser)

    if args.nfile is not None:
        process_neighbours(args)
        return

    datum = proto.Datum()
    env = annfab.utils.open_database(args.data)

    with env.begin() as txn:
        cursor = txn.cursor()
        if args.key:
            k = cursor.set_key(args.key)
            assert k, 'key ' + key + ' does not exist'
            key, value = cursor.item()
            datum.ParseFromString(value)
            show_image(key, datum)
            return
        iter(cursor)
        for key, value in cursor:
            datum.ParseFromString(value)
            show_image(key, datum)

            if not args.all:
                break


if __name__ == "__main__":
    main()
