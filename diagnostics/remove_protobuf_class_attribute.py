# ******************************************************************************
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import argparse
import errno
import os
import sys
"""
Removes the `_class` attribute from protobuffer nodes in pbtxts

usage: remove_protobuf_class_attribute.py [-h] (-f FILE | -d DIRECTORY)
                                           [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  pbtxt from tensorflow
  -d DIRECTORY, --directory DIRECTORY
                        directory of pbtxts to modify
  -o OUTPUT, --output OUTPUT
                        Output file or directory. If a single pbtxt was
                        provided as input (-f), provide here an output
                        filename. If a directory of pbtxts was provided (-d),
                        provide here an output directory path.

Examples:

# Provide a single input pbtxt
python remove_protobuf_class_attribute.py -f ./path/precapture_0000.pbtxt -o temp.pbtxt

# Provide a directory of input pbtxts
python remove_protobuf_class_attribute.py -d /path/to/pbtxts/ -o /path/to/desired/output/directory/

# Provide a single input pbtxt to modify in place
python remove_protobuf_class_attribute.py -f ./path/precapture_0000.pbtxt


"""


def get_files(directory):
    files = os.listdir(path=directory)
    pbtxts = []
    for f in files:
        if ".pbtxt" in f:
            pbtxts.append(f)
    return pbtxts


def main():
    """
    Remove the `_class` attribute from a pbtxt node.
    """
    args = get_args()
    if args.directory:
        files = get_files(args.directory)
        if args.output == None:
            # in place modification
            args.output = args.directory
        else:
            # non-destructive output directory
            try:
                os.makedirs(args.output)
            except OSError as exc:
                if exc.errno == errno.EEXIST and os.path.isdir(args.output):
                    pass
    else:
        if args.output == None:
            # in place modification
            args.output = args.file
        files = [args.file]
    for f in files:
        output_file = args.output
        if args.directory:
            output_file += "/" + f
        sys.stdout.write(progress("Processing: " + output_file))
        lines = []
        if args.directory:
            f = args.directory + "/" + f
        for line in open(f):
            lines.append(line)
        pruned = []
        save = True
        blacklist = 0
        for i, line in enumerate(lines):
            if "_class" in line:
                del pruned[-1]
                blacklist = i + 6
                save = False
            if i > blacklist:
                save = True
            if save:
                pruned.append(line)
        with open(output_file, "w") as output:
            for line in pruned:
                output.write(line)


class term_colors:
    OKBLUE = '\033[94m'
    ENDC = '\033[0m'


def progress(result):
    return term_colors.OKBLUE + result + term_colors.ENDC + "\r"


def get_args():
    """
    Argument parser initialization
    """
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-f", "--file", help="pbtxt from tensorflow", default=None)
    group.add_argument(
        "-d", "--directory", help="directory of pbtxts to modify", default=None)
    parser.add_argument(
        "-o",
        "--output",
        help=
        "Output file or directory. If a single pbtxt was provided as input (-f), provide here an output filename. If a directory of pbtxts was provided (-d), provide here an output directory path.",
        default=None)

    return parser.parse_args()


if __name__ == "__main__":
    main()
