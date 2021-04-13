# ==============================================================================
# Copyright (C) 2021 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
from platform import system
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel
import os
import sys

# https://stackoverflow.com/questions/45150304/how-to-force-a-python-wheel-to-be-platform-specific-when-building-it
class BinaryBdistWheel(bdist_wheel):
    def finalize_options(self):
        # bdist_wheel is old-style class in python 2, so can't `super`
        bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = bdist_wheel.get_tag(self)
        if system() == 'Linux':
           plat = 'linux_x86_64'
        py_version = 'py{}{}'.format(sys.version_info[0], sys.version_info[1])
        return (py_version, 'none', plat)

ext = 'dylib' if system() == 'Darwin' else 'so'

with open(@README_DOC@, "r", encoding="utf-8") as fh:
    long_description = fh.read()

# The following is filled in my cmake - essentially a list of library
# and license files
ng_data_list = [
    @ovtf_libraries@ @license_files@ @licence_top_level@
]

# This is the contents of the Package Data
package_data_dict = {}
package_data_dict['openvino_tensorflow'] = ng_data_list

import tensorflow as tf
tf_version = "tensorflow==" + tf.__version__

setup(
    name='openvino_tensorflow',
    version='0.5.0',
    description='Intel OpenVINO™ integration with TensorFlow',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/openvinotoolkit/openvino_tensorflow',
    packages=['openvino_tensorflow'],
    author='Intel Corporation',
    license='Apache License, Version 2.0',
    platforms='Ubuntu 18.04',
    include_package_data=True,
    package_data= package_data_dict,
    cmdclass={'bdist_wheel': BinaryBdistWheel},
    install_requires=[
        #[TODO] Replace this with custom built TF
        #tf_version,
    ],
)
