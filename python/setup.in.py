# ==============================================================================
# Copyright (C) 2021-2022 Intel Corporation
 
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
from platform import system
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from wheel.bdist_wheel import bdist_wheel
from wheel.vendored.packaging.tags import sys_tags
from pathlib import Path
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
            # Get the right platform tag by querying the linker version
            glibc_major, glibc_minor = os.popen("ldd --version | head -1").read().split()[-1].split(".")
            # OVTF is built against GLIBC 2.17 with ABI-0 for compatibility with TensorFlow wheels
            # See https://github.com/mayeut/pep600_compliance/blob/master/pep600_compliance/tools/manylinux-policy.json
            if glibc_major == "2" and glibc_minor == "17":
                plat = 'manylinux_2_17_x86_64.manylinux2014_x86_64'
            else: # For manylinux2014 and above, no alias is required
                plat = 'manylinux_%s_%s_x86_64'%(glibc_major, glibc_minor)
        tags = next(sys_tags())
        return (tags.interpreter, tags.abi, plat)

class InstallCommand(InstallCommandBase):

  def finalize_options(self):
    ret = InstallCommandBase.finalize_options(self)
    self.install_lib = self.install_platlib
    return ret

ext = 'dylib' if system() == 'Darwin' else 'so'

with open(@README_DOC@, "r", encoding="utf-8") as fh:
    long_description = fh.read()

# The following is filled in my cmake - essentially a list of library
# and license files
ov_libs_list = [
    @ovtf_libraries@ @license_files@ @licence_top_level@
]

# from https://github.com/openvinotoolkit/openvino/pull/13725
# transform libX.so.Y / libX.Y.dylib symlinks to real files
for symlink in Path(".").rglob("*"):
    if symlink.is_symlink():
        file_name = os.readlink(symlink)
        if not os.path.isabs(file_name):
            file_name = os.path.join(os.path.dirname(symlink), file_name)

        os.unlink(symlink)
        os.rename(file_name, symlink)
        print(f"Resolved symlink {symlink} as {file_name}")

# This is the contents of the Package Data
package_data_dict = {}
package_data_dict['openvino_tensorflow'] = ov_libs_list

setup(
    name='openvino_tensorflow',
    version='2.2.0',
    description='OpenVINO™ integration with TensorFlow',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/openvinotoolkit/openvino_tensorflow',
    packages=['openvino_tensorflow'],
    author='Intel Corporation',
    license='Apache License, Version 2.0',
    platforms='Ubuntu 18.04',
    include_package_data=True,
    package_data= package_data_dict,
    cmdclass={
        'bdist_wheel': BinaryBdistWheel,
        'install': InstallCommand
    },
    install_requires=[
        #[TODO] Replace this with custom built TF
        #tf_version,
    ],
)
