#!/usr/bin/env python3

from rosbag import Bag
import joblib
import os
from glob import iglob

from smartargparse import parse_args

from dataset_generator_for_simple_abstrelposnet import (ConfigForSimpleAbstRelPosNet,
        DatasetGeneratorForSimpleAbstRelPosNet)

def generate_dataset(config, bagfile_path, bag_id) -> None:
    bag = Bag(bagfile_path)
    DatasetGeneratorForSimpleAbstRelPosNet(config, bag, bag_id=bag_id)()
    
def main() -> None:
    config = parse_args(ConfigForSimpleAbstRelPosNet)
    joblib.Parallel(n_jobs=-1, verbose=50)(
            joblib.delayed(generate_dataset)(config, bagfile_path, i) \
                    for i, bagfile_path in enumerate(iglob(os.path.join(config.bagfiles_dir, "*")))
            )

if __name__ == "__main__":
    main()
