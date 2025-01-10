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

This step will also quantize all parameters from double precision float (f64) to half integer (i16).

The `.bin` file format contains the following fields:

- 32 bytes for header information
    - Byte 0: header version number
        - For backwards compatibility, avoid using 0x41 / 'A' here
    - Byte 1-2: number of neurons in hidden layer (u16)
    - Bytes 3-31: reserved for future use
- Architecture name (model's shape identifier / version)
- A single ESC ('0x1b') character to end the architecture name
- A test value that is the result of passing an all-ones input to the model.
- A value representing the K sigmoid constant.
- A single contiguous array of all weights/parameters.

It is the engine code's responsibility to know exactly how many parameters there
are per layer, the order of the parameters, the nature of the layers, and how
many layers there are. The engine only works for a specific architecture, and it
will refuse to read the parameters for another one (based on the Architecture
field).

All data is stored little-endian.

As of writing, all data should be 64-bit floating point data (8 bytes per parameter).
"""

import numpy as np
import logging
import argparse

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


# quantization type
# `<` means little endian
# `ix` means int with x bytes
# `fx` means float with x bytes
# e.g. "<f2" is a half precision floating point
dtype = "<i2"

# quantization scaling factors
SCALE_L1 = 255
SCALE_OUT = 64


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

        # hidden layer size
        l1_size = min(model.l1_size, 0xffff)

        arch_specific: str = (model.arch_specific or arch) + "QT" + dtype

        arch += "QT" + dtype

        bin = args.output or args.pth.with_suffix(".bin")

        if bin.is_file():
            logging.error("File '%s' already exists; not overwriting.", bin)
            exit(1)

        all_ones_res = np.double(
            (
                model.linear_relu_stack(torch.ones([s3nn.INPUT_SIZE], dtype=torch.double))
                * model.k
            ).item()
        ).astype(dtype, casting="unsafe")

        print(f"writing architecture {arch_specific}")
        print(f"sanity check value: {all_ones_res}")
        print(
            "running the model with an all ones input should give you the above centipawn (logit) value."
        )
        print(
            "the logit value is for double precision float. results may vary when quantized."
        )
        print(
            "please be careful of endianness; this .bin file stores in little-endian."
        )

        k = np.ascontiguousarray(np.array(model.k), dtype=dtype)

        print(f"\nk is {k}")

        HEADER_VERSION = np.uint8(0)
        hidden_size = np.ascontiguousarray(np.array(l1_size), "<u2")
        PADDING = np.ascontiguousarray(np.zeros(29), "<u1")

        with open(bin, "wb") as f:
            f.write(HEADER_VERSION)
            f.write(hidden_size)
            f.write(PADDING)
            f.write(arch.encode())
            f.write(b"\x1b")
            f.write(all_ones_res)
            f.write(k)
            params = list(model.parameters())
            params[0].data *= SCALE_L1
            # transpose l1 weights for efficiency
            params[0].data = torch.transpose(params[0].data, 0, 1)
            params[1].data *= SCALE_L1
            params[2].data *= SCALE_OUT
            params[3].data *= SCALE_OUT
            for param in params:
                f.write(params_bytes(param))
