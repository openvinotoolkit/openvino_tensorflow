# ******************************************************************************
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

from __future__ import print_function

import pickle as pkl
import os
import sys
import re
import pdb
from ngtf_graph_viewer import load_file


def create_node_encapsulate_map_pkl(input_dir, output_pkl_name):
    start_with = "declustered_"
    ends_with = ".pbtxt"
    pattern = re.compile("^" + start_with + "(.*?)" + ends_with + "$")
    # Note: relying on this particular pattern. Could be brittle if the filenames change
    declustered_pbtxts = filter(lambda file_name: pattern.search(file_name),
                                os.listdir(input_dir))
    node_cluster_map = {}
    for filename in declustered_pbtxts:
        full_name = os.path.join(input_dir, filename)
        print('Reading: ' + filename)
        gdef = load_file(full_name, input_binary=False)
        print('Processing: ' + filename)
        for idx, node in enumerate(gdef.node):
            if '_ngraph_cluster' in node.attr:
                node_cluster_map[node.name] = 'ngtf_' + \
                    str(node.attr['_ngraph_cluster'].i)+'/'
    pkl.dump(node_cluster_map, open(output_pkl_name, "wb"), protocol=2)


if __name__ == '__main__':
    create_node_encapsulate_map_pkl(sys.argv[1], sys.argv[2])
    # python get_node_encapsulate_map.py /path/to/where/the/declustered/pbtxts/were/dumped nodemap.pkl
