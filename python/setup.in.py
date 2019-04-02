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

setup( 
    name='ngraph_tensorflow_bridge',
    version='0.12.0rc2',
    description='Intel nGraph compiler and runtime for TensorFlow',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/NervanaSystems/ngraph-tf/',
    packages=['ngraph_bridge'], 
    author='Intel Nervana', 
    license='Apache License, Version 2.0',
    platforms='Ubuntu 16.04, macOS Sierra',
    include_package_data=True,
    package_data=
    {
        'ngraph_bridge': [
            @ngraph_libraries@ @license_files@ @licence_top_level@
        ],
    },
    cmdclass={'bdist_wheel': BinaryBdistWheel},
    extras_require={
        'plaidml': ["plaidml>=0.5.0"],
    },
)
