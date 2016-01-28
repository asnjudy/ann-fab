# Generate an ANN structure for a given data set.

import os
import cProfile
import argparse
import logging
import sys

import config

import annfab.storage
import annfab.hasher
import annfab.utils
try:
    import annfab.batch_hasher
    disable_batch = False
except ImportError as e:
    logging.error("Cannot import annfab.batch_hasher: %s" % e)
    logging.error("Disabling batch mode")
    disable_batch = True


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate a nearest neighbor filter engine")

    parser.add_argument("--outfile",
                        default=None,
                        help="The filename to use to save the output.")
    parser.add_argument("-L",
                        type=int,
                        default=550,
                        help="The number of hashes to use.")
    parser.add_argument("-m",
                        type=int,
                        default=8,
                        help="The dimension of each hash")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")

    return parser


def check_args(args):
    if args.batch_size > 1 and disable_batch:
        logging.error("Batching is disabled. Please select a batch size of 1")
        return False
    if args.batch_size == 1 and args.gpu:
        logging.error("GPU mode is only supported for batched operation.")
        return False

    return True


def generate_model():
    parser = create_arg_parser()
    args = annfab.utils.parse_command_line(parser)

    if not check_args(args):
        sys.exit()

    # Open the LMDB data base.
    db_name = os.path.basename(args.data)
    env = annfab.utils.open_database(args.data)

    # Create the LMDB storage backend
    storage = annfab.storage.LmdbStorage(
        env, annfab.utils.normalized_image_vector)

    if disable_batch and args.batch_size != 1:
        logging.info("Batch mode disabled. \
                      Using a batch size of 1 and not %d" % args.batch_size)
        args.batch_size = 1

    if args.batch_size == 1:
        # Create an image hasher.
        hasher = annfab.hasher.Hasher(
            storage, db_name, args.L, args.m)
    else:
        hasher = annfab.batch_hasher.BatchHasher(
            storage, db_name, args.L, args.m, args.gpu)
        hasher.set_batch_size(args.batch_size)

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    with env.begin() as txn:
        cursor = txn.cursor()
        iter(cursor)
        for key, value in cursor:
            hasher.add_image(key, annfab.utils.value_to_image(value))

    if args.batch_size != 1:
        hasher.flush()

    hasher.save_to_file(args.outfile)

    if args.profile:
        pr.create_stats()
        pr.print_stats()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_model()
