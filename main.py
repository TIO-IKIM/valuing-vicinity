# call in terminal, e.g.:
# python main.py --conf_file configs_paper/configs_cy16/attention/deeplab_resnet18/a_d2_k8_deeplab_no_help_config.yml --gpu 4

import argparse
import sys
import traceback

from src.exp_management.experiment.SegmentationExperiment import SegmentationExperiment
from src.exp_management.run_experiment import run_experiment

if __name__ == '__main__':

    conf_parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False
        )
    
    conf_parser.add_argument("-c", "--conf_file",
                        help="Specify config file", metavar="FILE")
    # unkown args are ignoed but might be passed by experiment config parsers
    args, _ = conf_parser.parse_known_args()
    
    try:
        run_experiment(exp=SegmentationExperiment(config_path=args.conf_file))
    except Exception as e:
        print(traceback.format_exc())
        sys.exit(e)                     


