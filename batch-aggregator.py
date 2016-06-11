import itertools
import logging
import argparse
import glob
import os
import numpy as np

logger = logging.getLogger(__name__)


def processing(input_folder, output):
    files = glob.glob(os.path.join(input_folder, "*"))
    tmp_fname = output + "_tmp"
    logger.info("Start first stage (collect all probs)")
    with open(tmp_fname, 'w') as outfile:
        for idx, fl in enumerate(files):
            logger.info("Iter #{}/{}, file {}".format(idx + 1, len(files), fl))
            with open(fl, 'r') as infile:
                for line in infile:
                    _, prob, lab, d_list, in_train = line.strip().split("\t")
                    lab, in_train = int(lab), int(in_train)
                    d_list = d_list.split(",")
                    if lab == 1 and in_train != -1:
                        for d in d_list:
                            if d.endswith('in-addr.arpa') or \
                               d.endswith('sophosxl.net') or \
                               d.endswith('webcfs00.com') or \
                               d.endswith('loudtalks.com') or \
                               d.startswith('yahoo.'):
                                continue
                            outfile.write("{}\t{}\n".format(d, prob))
    logger.info("Start second stage mean(prob)")
    os.system("sort {0} -o {0}".format(tmp_fname))
    with open(tmp_fname, 'r') as infile, open(output, 'w') as outfile:
        for k, data in itertools.groupby(infile, key=lambda _: _.split('\t')[0]):
            scores = [float(x.strip().split('\t')[1]) for x in data]
            if np.mean(scores) > 0.52 and len(scores) > 2:
                outfile.write("{}\n".format(k))
    logger.info("Complete aggregation")


def main():
    parser = argparse.ArgumentParser(description="Aggregate files to final output for gb")
    parser.add_argument('--folder', help='Path to folder with output', type=str, required=True)
    parser.add_argument('--output', help='Path to outfile', type=str, required=True)
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    return processing(args.folder, args.output)


if __name__ == '__main__':
    exit(main())
