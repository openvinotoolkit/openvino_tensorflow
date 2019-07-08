#==============================================================================
#  Copyright 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# =============================================================================
import argparse
import os


def transform(input, output, print_header=False):
    # Datafile format that we are parsing
    # ...
    # TEST: run-dcgan.sh
    # PASSED
    # RESULT: batch size is 100
    # RESULT: latency is 34.451300 (ms)
    # RESULT: throughput is 2886.000000 (samples/second)
    # --------------------------------------------
    # TEST: run-densenet.sh
    # FAILED
    # RESULT: batch size is 0
    # RESULT: latency is nan
    # RESULT: throughput is nan
    # --------------------------------------------
    # ...
    with open(input, 'r') as in_fp:
        out_fp = open(output, 'a')
        header_line = ""
        value_line = ""
        for line in in_fp:
            line = line.strip()
            if 'TEST' in line:
                # Read until we get the next '-----------...'
                lines = []
                lines.append(line.split(':')[1].strip())
                while line:
                    line = in_fp.readline().strip()
                    lines.append(line)
                    if '---------------------------------' in line:
                        # Check if the result indicate pass or fail
                        break
                if 'PASSED' in lines[1]:
                    # test file name example:
                    # run-wide-and-deep.sh
                    test_name = lines[0].split(':')[0].split('.')[0].strip()
                    test_name = test_name.split('run-')[1]
                    throughput = lines[4].split(' ')[3].strip()
                    value_line += (throughput + ' ')
                    if print_header:
                        header_line += (test_name + ' ')
                    print('Result: %s Throughput: %s' % (test_name, throughput))
        if print_header:
            out_fp.write(header_line + '\n')
        out_fp.write(value_line)
        out_fp.write('\n')
        out_fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '--input_log',
        type=str,
        help="Log file to process\n",
        action="store",
        required=True)
    parser.add_argument(
        '--output_file',
        type=str,
        help="Name of the output file the parsed output will be appended to",
        action="store",
        required=True)

    arguments = parser.parse_args()

    # If the output file doesn't exist - then write the header
    print_header = False
    if not os.path.isfile(arguments.output_file):
        print_header = True
    transform(arguments.input_log, arguments.output_file, print_header)
