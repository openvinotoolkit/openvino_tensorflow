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
from setuptools.dist import Distribution

ext = 'dylib' if system() == 'Darwin' else 'so'

class BinaryDistribution(Distribution):
    def is_pure(self):
        return False

setup( 
    name='ngraph',
    version='0.0.0',
    description='Intel nGraph device',
    packages=['ngraph'], 
    author='Intel-Nervana AIPG', 
    include_package_data=True,
    distclass=BinaryDistribution,
    package_data={
            'ngraph': [@ngraph_libraries@],
                },
)

