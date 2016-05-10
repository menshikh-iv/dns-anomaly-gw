import logging
import argparse
import glob
import os

logger = logging.getLogger(__name__)


def processing(input_folder, output):
    files = glob.glob(os.path.join(input_folder, "*"))
    with open(output, 'w') as outfile:
        for idx, fl in enumerate(files):
            logger.info("Iter #{}/{}, file {}".format(idx + 1, len(files), fl))
            with open(fl, 'r') as infile:
                for line in infile:
                    _, prob, lab, d_list, in_train = line.strip().split("\t")
                    lab, in_train = int(lab), int(in_train)
                    d_list = d_list.split(",")
                    if lab == 1 and in_train != -1:
                        for d in d_list:
                            if d.endswith('in-addr.arpa'):
                                continue
                            outfile.write("{}\n".format(d))


def main():
    parser = argparse.ArgumentParser(description="Aggregate files to final output for gb")
    parser.add_argument('--folder', help='Path to folder with output', type=str, required=True)
    parser.add_argument('--output', help='Path to outfile', type=str, required=True)
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    return processing(args.folder, args.output)


if __name__ == '__main__':
    exit(main())
