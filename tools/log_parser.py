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


def parse_logs(log_lines):
    if type(log_lines) == type(''):
        log_lines = log_lines.split('\n')
    else:
        assert type(log_lines) == type(
            []
        ), "If log_lines if not a string, it should have been a list, but instead it is a " + type(
            log_lines)
    assert all([
        type(i) == type('') and '\n' not in i for i in log_lines
    ]), 'Each element of the list should be a string and not contain new lines'
    all_results = {}
    curr_result = {}
    ctr = 0
    for line in log_lines:
        start_of_subgraph = "NGTF_SUMMARY: Op_not_supported:" in line
        # If logs of a new sub-graph is starting, save the old one
        if start_of_subgraph:
            if len(curr_result) > 0:
                all_results[str(ctr)] = curr_result
                curr_result = {}
                ctr += 1
        # keep collecting information in curr_result
        if line.startswith('NGTF_SUMMARY'):
            if 'Number of nodes in the graph' in line:
                curr_result['num_nodes_in_graph'] = int(
                    line.split(':')[-1].strip())
            elif 'Number of nodes marked for clustering' in line:
                curr_result['num_nodes_marked_for_clustering'] = int(
                    line.split(':')[-1].strip().split(' ')[0].strip())
            elif 'Number of ngraph clusters' in line:
                curr_result['num_ng_clusters'] = int(
                    line.split(':')[-1].strip())
            # TODO: fill other information as needed

    # add the last subgraph to all_results
    all_results[str(ctr)] = curr_result

    return all_results


def compare_parsed_values(parsed_vals, expected_vals):
    # Both inputs are expected to be 2 dictionaries (representing jsons)
    # The constraints in expected is <= parsed_vals. Parsed_vals should have all possible values that the parser can spit out. However expected_vals can be relaxed (even empty) and choose to only verify/match certain fields
    match = lambda current, expected: all(
        [expected[k] == current[k] for k in expected])
    for graph_id_1 in expected_vals:
        # The ordering is not important and could be different, hence search through all elements of parsed_vals
        matching_id = None
        for graph_id_2 in parsed_vals:
            if match(expected_vals[graph_id_1], parsed_vals[graph_id_2]):
                matching_id = graph_id_2
                break
        if matching_id is None:
            return False, 'Failed to match expected graph info ' + graph_id_1 + " which was: " + str(
                expected_vals[graph_id_1]
            ) + "\n. Got the following parsed results: " + str(parsed_vals)
        else:
            parsed_vals.pop(matching_id)
    return True, ''
