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
        return ('py2.py3', 'none', plat)

ext = 'dylib' if system() == 'Darwin' else 'so'

setup( 
    name='ngraph_config',
    version='0.8.0',
    description='Intel nGraph compiler and runtime',
    url='https://ai.intel.com/intel-ngraph/',
    packages=['ngraph_config'], 
    author='Intel Nervana', 
    license='Apache License, Version 2.0',
    platforms='Ubuntu 16.04, macOS Sierra',
    include_package_data=True,
    package_data={
        'ngraph_config': [@ngraph_libraries@],
                 },
    cmdclass={'bdist_wheel': BinaryBdistWheel},
)
