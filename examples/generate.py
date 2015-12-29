# Generate an ANN structure for a given data set.

import os
import cProfile
import argparse

import annfab.storage
import annfab.hasher
try:
    import annfab_examples.batch_hasher
except:
    pass
import annfab.utils


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="Show images in an lmdb database")

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

    return parser


def generate_model():
    parser = create_arg_parser()
    args = annfab.utils.parse_command_line(parser)

    # Open the LMDB data base.
    db_name = os.path.basename(args.data)
    env = annfab.utils.open_database(args.data)

    # Create the LMDB storage backend
    storage = annfab.storage.LmdbStorage(
        env, annfab.utils.normalized_image_vector)

    if args.batch_size == 1:
        # Create an image hasher.
        hasher = annfab_examples.hasher.Hasher(
            storage, db_name, args.L, args.m)
    else:
        hasher = annfab_examples.batch_hasher.BatchHasher(
            storage, db_name, args.L, args.m)
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
    generate_model()
