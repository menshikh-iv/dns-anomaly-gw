import logging
import argparse
import json
import glob
import os

MUST_BE_IN_CONFIG = {"whitelist", "blacklist", "n_folds", "n_iter"}
logger = logging.getLogger(__name__)


def processing(input_folder, output_folder, conf_path, batch_size):
    with open(conf_path, 'r') as infile:
        conf = json.load(infile)

    missed_params = MUST_BE_IN_CONFIG - set(conf.keys())
    if len(missed_params):
        raise RuntimeError("Define variables {} in config".format(','.join(missed_params)))

    if not os.path.exists(output_folder):
        logger.info("Create folder %s", output_folder)
        os.makedirs(output_folder)

    input_files = glob.glob(os.path.join(input_folder, "*"))
    input_files.sort()

    batches = [input_files[idx: idx + batch_size] for idx in range(0, len(input_files), batch_size)]
    if len(batches[-1]) < batch_size:
        logger.warning("Skip last batch %s", ",".join(batches.pop()))

    for idx, b in enumerate(batches):
        for_call = ["python suspicious-detection.py",
                    "--files {}".format(" ".join(b)),
                    "--blacklist {}".format(conf["blacklist"]),
                    "--whitelist {}".format(conf["whitelist"]),
                    "--output {}".format(os.path.join(output_folder, "-".join(b))),
                    "--n_iter {}".format(conf["n_iter"]),
                    "--n_folds {}".format(conf["n_folds"]),
                    "--verbose"]

        logger.info("Processing bath #%d of %d", idx + 1, len(batches))
        result_string = " ".join(for_call)
        logger.info("Run %s", result_string)
        os.system(result_string)
        logger.info("=" * 20)


def main():
    parser = argparse.ArgumentParser(description="Batch runner for suspicious-detection.py")
    parser.add_argument('--input', help='Path to folder with logs', type=str, required=True)
    parser.add_argument('--output', help='Path to folder for output', type=str, required=True)
    parser.add_argument('--conf', help='Path to configuration file (json)', type=str, required=True)
    parser.add_argument('--batch_size', help='Size of batch', type=int, default=3)

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    return processing(args.input, args.output, args.conf, args.batch_size)


if __name__ == '__main__':
    exit(main())
