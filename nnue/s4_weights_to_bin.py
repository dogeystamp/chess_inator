#!/usr/bin/env python

# This file is part of chess_inator.
# chess_inator is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.
#
# chess_inator is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with chess_inator. If not, see https://www.gnu.org/licenses/.
#
# Copyright © 2025 dogeystamp <dogeystamp@disroot.org>

"""
Convert PyTorch `.pth` weights to `.bin` weights.

This step will also quantize all parameters from double to single precision float.

The `.bin` file format contains the following fields:

- Architecture name (model's shape identifier / version)
- A single ESC ('0x1b') character to end the architecture name
- A test value that is the result of passing an all-ones input to the model.
- A single contiguous array of all weights/parameters.

It is the engine code's responsibility to know exactly how many parameters there
are per layer, the order of the parameters, the nature of the layers, and how
many layers there are. The engine only works for a specific architecture, and it
will refuse to read the parameters for another one (based on the Architecture
field).

All data is stored little-endian.

As of writing, all data should be 64-bit floating point data (8 bytes per parameter).
"""

from typing import Iterator
import numpy as np
import logging
import argparse
import functools
import operator

import torch
import s3_train_neural_net as s3nn

from pathlib import Path
from sys import exit


logging.basicConfig(level=logging.INFO)


################################
################################
## CLI argument parsing
################################
################################

parser = argparse.ArgumentParser()
parser.add_argument(
    "pth",
    type=Path,
    help="Weights in PyTorch .pth format.",
    default=Path("weights.pth"),
)
parser.add_argument(
    "-o",
    "--output",
    type=Path,
    help="Output path for .bin format weights.",
)

################################
################################
## main logic
################################
################################


# quantize to single precision float (little endian)
dtype = "<f4"

def params_bytes(obj) -> bytes:
    """Convert objects into bytes."""

    global dtype
    return np.ascontiguousarray(obj, dtype=dtype).tobytes()


if __name__ == "__main__":
    args = parser.parse_args()

    with torch.no_grad():
        model = s3nn.NNUE()
        s3nn.load_model(args.pth, model, None)

        arch: str = model.arch

        arch += "_q" + dtype

        bin = args.output or args.pth.with_suffix(".bin")

        if bin.is_file():
            logging.error("File '%s' already exists; not overwriting.", bin)
            exit(1)

        all_ones_res = np.double(
            (
                model.linear_relu_stack(torch.ones([s3nn.INPUT_SIZE], dtype=torch.double))
            ).item()
        ).astype(dtype, casting="same_kind")

        print(f"sanity check value: {all_ones_res}")
        print(
            "running the model with an all ones input should give you the above logit value (i.e. before the sigmoid)."
        )
        print(
            "please be careful of endianness; this .bin file stores in little-endian."
        )

        with open(bin, "wb") as f:
            f.write(arch.encode())
            f.write(b"\x1b")
            f.write(all_ones_res)
            for param in model.parameters():
                f.write(params_bytes(param))
