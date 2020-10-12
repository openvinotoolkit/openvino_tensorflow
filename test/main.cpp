/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include <dlfcn.h>
#include <stdlib.h>
#include <array>
#include <cctype>
#include <chrono>
#include <clocale>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>

#include "gtest/gtest.h"

#include "tensorflow/core/platform/env.h"

#ifdef __APPLE__
#define EXT "dylib"
#else
#define EXT "so"
#endif

using namespace std;

string SCRIPTDIR =
    std::regex_replace(__FILE__, std::regex("^(.*)/[^/]+$"), "$1");

static string str_tolower(string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

template <class Set1, class Set2>
bool is_disjoint(const Set1& set1, const Set2& set2) {
  if (set1.empty() || set2.empty()) return true;

  typename Set1::const_iterator it1 = set1.begin(), it1End = set1.end();
  typename Set2::const_iterator it2 = set2.begin(), it2End = set2.end();

  if (*it1 > *set2.rbegin() || *it2 > *set1.rbegin()) return true;

  while (it1 != it1End && it2 != it2End) {
    if (*it1 == *it2) return false;
    if (*it1 < *it2) {
      it1++;
    } else {
      it2++;
    }
  }

  return true;
}

void set_diff(set<string>& set1, const set<string>& set2) {
  for (const auto& elem : set2) {
    set1.erase(elem);
  }
}

// manifestfile must be in abs path format
static void read_tests_from_manifest(string manifestfile, set<string>& run_list,
                                     set<string>& skip_list) {
  static set<string> g_imported_files;
  fstream fs;
  fs.open(manifestfile, ios::in);
  if (fs.is_open()) {
    cout << "Parsing manifest: " << manifestfile << " ...\n";
    string line;
    string curr_section = "";
    g_imported_files.insert(manifestfile);
    while (getline(fs, line)) {
      line = std::regex_replace(line, std::regex("^\\s+"), "");
      line = std::regex_replace(line, std::regex("#.*$"), "");
      line = std::regex_replace(line, std::regex("\\s+$"), "");
      if (line.empty()) continue;
      if (std::regex_search(line, std::regex("^\\[IMPORT\\]$"))) {
        curr_section = "import_section";
        continue;
      }
      if (std::regex_search(line, std::regex("^\\[RUN\\]$"))) {
        curr_section = "run_section";
        continue;
      }
      if (std::regex_search(line, std::regex("^\\[SKIP\\]$"))) {
        curr_section = "skip_section";
        continue;
      }
      if (curr_section == "import_section") {
        if (g_imported_files.find(line) != g_imported_files.end()) {
          cout << "ERROR: re-import of manifest " << line << " in "
               << manifestfile << endl;
          exit(1);
        }
        line = SCRIPTDIR + "/" + line;
        g_imported_files.insert(line);
        set<string> new_runs, new_skips;
        read_tests_from_manifest(line, new_runs, new_skips);
        assert((is_disjoint<set<string>, set<string>>(new_runs, new_skips)));
        run_list.insert(new_runs.begin(), new_runs.end());
        skip_list.insert(new_skips.begin(), new_skips.end());
        set_diff(run_list, skip_list);
      }
      if (std::regex_search(line, std::regex("[:\\s]"))) {
        cout << "Bad pattern: [" << line << "], ignoring...\n";
        continue;
      }
      if (curr_section == "run_section") {
        skip_list.erase(line);
        run_list.insert(line);
      }
      if (curr_section == "skip_section") {
        run_list.erase(line);
        skip_list.insert(line);
      }
    }
    fs.close();
    assert((is_disjoint<set<string>, set<string>>(run_list, skip_list)));
  } else {
    cout << "Cannot open file: <" << manifestfile << ">\n";
  }
}

class TestEnv {
 public:
  static string get_platform_type() {
// 'Linux', 'Windows', 'Darwin', or 'Unknown'
#ifdef _WIN32
    return "Windows";
#elif __APPLE__ || __MACH__
    return "Darwin";
#elif __linux__
    return "Linux";
#else
    return "Unknown";
#endif
  }

  static string get_test_manifest_filename() {
    char* env = getenv("NGRAPH_TF_TEST_MANIFEST");
    if (env) {
      return std::string(env);
    }
    // test manifest files are named like this:
    // tests_${PLATFORM}_${NGRAPH_TF_BACKEND}.txt
    return string("tests_") + str_tolower(TestEnv::PLATFORM()) + string("_") +
           str_tolower(TestEnv::BACKEND()) + string(".txt");
  }

  static string PLATFORM() { return get_platform_type(); }

  static string BACKEND() {
    char* env = getenv("NGRAPH_TF_BACKEND");
    if (env) {
      return std::string(env);
    }
    return "CPU";
  }
};

static string get_test_manifest_filepath() {
  string str = TestEnv::get_test_manifest_filename();
  if (str.at(0) != '/') {
    str = SCRIPTDIR + "/" + str;
  }
  return str;
}

static void set_filters_from_file() {
  string filter_file = get_test_manifest_filepath();
  set<string> run_list, skip_list;
  read_tests_from_manifest(filter_file, run_list, skip_list);

  string filters = "";
  for (auto& it : run_list) {
    filters += it + ":";
  }
  if (filters.back() == ':') filters.pop_back();  // remove last :
  if (skip_list.size() > 0)
    filters += "-";  // separator before the skips/excludes
  for (auto& it : skip_list) {
    filters += it + ":";
  }
  if (filters.back() == ':') filters.pop_back();  // remove last :

  ::testing::GTEST_FLAG(filter) = filters;
}

bool is_arg_provided(int argc, char** argv, string which) {
  bool provided = false;
  for (int i = 1; i < argc; i++) {
    string s(argv[i]);
    if (s.rfind(which, 0) == 0) {
      // s starts with prefix
      provided = true;
      break;
    }
  }
  return provided;
}

int main(int argc, char** argv) {
  bool filter_arg = is_arg_provided(argc, argv, "--gtest_filter");
  bool list_tests_arg = is_arg_provided(argc, argv, "--gtest_list_tests");

  if (list_tests_arg) {
    cout << "Checking test manifest for listing...\n";
    set_filters_from_file();
    cout << "\n--gtest_list_tests : " << ::testing::GTEST_FLAG(filter)
         << "\n\n";
  } else if (!filter_arg && argc == 1) {
    // user has not given any explicit filters
    cout << "Using test manifest to set test filters...\n";
    set_filters_from_file();
  }

  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  return rc;
}
