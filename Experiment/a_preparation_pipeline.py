# coding: utf-8

# ===================================================================
# Data preparation pipeline
# Romuald Menuet - May 2019 2018
# ===================================================================
# Summary: This script calls all the data management scripts sequentially
#          for a decoding meta-analysis of Neurovault fMRIs stat maps
# ===================================================================

import argparse
import json
from a1_collect import prepare_collect
from a2_filter import prepare_filter
from a3_resample import prepare_resample
from a4_mask import prepare_mask
from a5_embed import prepare_embed
from a6_impute import prepare_impute
from a7_label import prepare_label


if __name__ == "__main__":
    # execute only if run as a script
    parser = argparse.ArgumentParser(
        description="A full preparation pipeline to download and preprocess "
                    "data from Neurovault before a decoding analysis.",
        epilog="Example: "
               "python a_preparation_pipeline.py -C config.json -j 8 -v"
    )
    parser.add_argument("-C", "--configuration",
                        default="./config.json",
                        help="Path of the JSON configuration file")
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=1,
                        help="Number of jobs")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Activates (many) outputs")

    args = parser.parse_args()

    with open(args.configuration) as f:
        global_config = json.load(f)

    print("=== PREPARATION PIPELINE ===")

    print("=     DATA COLLECTION    =")
    prepare_collect(global_config, args.verbose)
    print("=    SAMPLES FILTERING   =")
    prepare_filter(global_config, args.jobs, args.verbose)
    print("=       RESAMPLING       =")
    prepare_resample(global_config, args.jobs, args.verbose)
    print("=         MASKING        =")
    prepare_mask(global_config, args.jobs, args.verbose)
    print("=        EMBEDDING       =")
    prepare_embed(global_config, args.verbose)
    print("=       IMPUTATION       =")
    prepare_impute(global_config, args.verbose)
    print("=    LABELS EXTRACTION   =")
    prepare_label(global_config, args.verbose)

    print("=== PREPARATION FINISHED ===")
