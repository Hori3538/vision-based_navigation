#!/usr/bin/env python3

import os
from argparse import ArgumentParser

import torch
from my_models import OrientationNet


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--weight-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    model = OrientationNet().eval().cuda()
    model.load_state_dict(torch.load(args.weight_path, map_location="cuda:0"))

    sample = torch.randn(1, 3, 224, 224).cuda(), torch.randn(1, 3, 224, 224).cuda()
    print("Tracing model.")
    traced = torch.jit.trace(model, sample)  # type: ignore
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    traced.save(args.output_path)  # type: ignore
    print(f"Saved torch script: {args.output_path}")

if __name__ == '__main__':
    main()
