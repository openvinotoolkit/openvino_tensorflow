#!/usr/bin/env python3
# ==============================================================================
#  Copyright 2018-2021 Intel Corporation
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
# ==============================================================================
import argparse
import errno
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv',
        type=str,
        help="Input CSV file for generating plot.\n",
        action="store")
    parser.add_argument(
        '--title',
        type=str,
        help="Optional title for plot.\n",
        default="Plot",
        action="store")
    parser.add_argument(
        '--ylabel',
        type=str,
        help="Optional ylabel for plot.\n",
        default="",
        action="store")

    arguments = parser.parse_args()

    # Check for mandetary parameters
    if not arguments.csv:
        raise Exception("Need to specify --csv")

    csv_filepath = os.path.abspath(arguments.csv)
    if (os.path.isfile(csv_filepath) == False):
        raise Exception("CSV file not found: " + csv_filepath)
    benchmark_data = pd.read_csv(csv_filepath)

    plt.figure(figsize=(16, 6))
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.3)
    plt.title(arguments.title)
    plt.ylabel(arguments.ylabel)
    width = 0.2
    xdata = benchmark_data['Model']
    xaxis = np.arange(len(xdata))
    bar_tf = plt.bar(
        xaxis,
        benchmark_data['Stock-TF'],
        width=width,
        color='lightgray',
        label="Stock-TF")
    bar_stockov = plt.bar(
        x=xaxis + width,
        height=benchmark_data['OV'],
        width=width,
        color='orange',
        label="OV")
    bar_tfov = plt.bar(
        x=xaxis + 2 * width,
        height=benchmark_data['TF-OV'],
        width=width,
        color='royalblue',
        label="TF-OV")

    plt.xticks(xaxis, xdata, rotation=90)  # Create names on the x-axis
    plt.legend(handles=[bar_tf, bar_stockov, bar_tfov])

    image_filename = os.path.splitext(csv_filepath)[0] + ".png"
    plt.savefig(image_filename)  # saved as <same-prefix-as-csv>.png


if __name__ == '__main__':
    main()
