# ==============================================================================
#  Copyright 2018 Intel Corporation
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
from platform import system
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel
import os

# https://stackoverflow.com/questions/45150304/how-to-force-a-python-wheel-to-be-platform-specific-when-building-it
class BinaryBdistWheel(bdist_wheel):
    def finalize_options(self):
        # bdist_wheel is old-style class in python 2, so can't `super`
        bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = bdist_wheel.get_tag(self)
        if system() == 'Linux':
           plat = 'manylinux1_x86_64'

        return ('py2.py3', 'none', plat)

ext = 'dylib' if system() == 'Darwin' else 'so'

with open(@README_DOC@, "r") as fh:
    long_description = fh.read()

# Collect the list of include files, while preserving the tree structure
os.chdir('ngraph_bridge')
include_list = []
for path, dirs, files in os.walk('include'):
  for f in files:
    include_list.append(path + "/" + f )

os.chdir('..')

# The following is filled in my cmake - essentially a list of library
# and license files
ng_data_list = [
    @ngraph_libraries@ @license_files@ @licence_top_level@
]
include_list.extend(ng_data_list)

# This is the contents of the Package Data
package_data_dict = {}
package_data_dict['ngraph_bridge'] = include_list

setup( 
    name='ngraph_tensorflow_bridge',
    version='0.12.0rc6',
    description='Intel nGraph compiler and runtime for TensorFlow',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/NervanaSystems/ngraph-tf/',
    packages=['ngraph_bridge'], 
    author='Intel Nervana', 
    license='Apache License, Version 2.0',
    platforms='Ubuntu 16.04, macOS Sierra',
    include_package_data=True,
    package_data= package_data_dict,
    cmdclass={'bdist_wheel': BinaryBdistWheel},
    extras_require={
        'plaidml': ["plaidml>=0.5.0"],
    },
)
