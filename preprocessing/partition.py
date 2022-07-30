import argparse
from pathlib import Path
import random


# Script to create train, test, validation splits.
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", required=True, type=str,
                        help="Path to dataset. Expects npz files in that folder")
    parser.add_argument("--out_dir", required=False, type=str, default="../examples",
                        help="Location where split files should be saved.")
    parser.add_argument("--ratio_test", default=0, type=float,
                        help='the ratio of files used for testing, will be normalized')
    parser.add_argument("--ratio_train", default=0.8, type=float,
                        help='the ratio of files used for training, will be normalized')
    parser.add_argument("--ratio_val", default=0.2, type=float,
                        help='the ratio of files used for validation, will be normalized')
    parser.add_argument("--seed", default=0, type=int,
                        help='the seed for the pseudo random number generator')
    args = parser.parse_args()

    # ensure validity of params
    args.data_dir = Path(args.data_dir)
    assert args.data_dir.exists(), 'data_dir does not exist: {}'.format(args.data_dir)

    ratio_sum = args.ratio_test + args.ratio_train + args.ratio_val
    assert abs(ratio_sum) > 1e-10, 'the given ratios are invalid'

    # normalize ratios
    args.ratio_test /= ratio_sum
    args.ratio_train /= ratio_sum
    args.ratio_val /= ratio_sum

    # get files
    file_list = [e.name for e in args.data_dir.iterdir() if e.is_file() and e.suffix == ".npz"]
    if len(file_list) == 0:
        assert False, "No npz files found. Wrong data directory."

    # create splits
    random.Random(args.seed).shuffle(file_list)
    n_test = int(round(args.ratio_test*len(file_list)))
    n_val = int(round(args.ratio_val*len(file_list)))
    # n_train = len(file_list) - n_test - n_val
    file_list_test = file_list[:n_test]
    file_list_val = file_list[n_test:(n_test+n_val)]
    file_list_train = file_list[(n_test+n_val):]

    # write files
    with open(f"{args.out_dir}/test_split_{args.data_dir.name}.txt", 'w') as f:
        f.writelines("%s\n" % item for item in file_list_test)
    with open(f"{args.out_dir}/val_split_{args.data_dir.name}.txt", 'w') as f:
        f.writelines("%s\n" % item for item in file_list_val)
    with open(f"{args.out_dir}/train_split_{args.data_dir.name}.txt", 'w') as f:
        f.writelines("%s\n" % item for item in file_list_train)


if __name__ == '__main__':
    main()
