Dockerfile Name                                        Image Characteristics                   Purpose
===============                                        =====================                   =======

Dockerfile.openvino_tf.build_ovtf_ubuntu1604_gcc48_py35  Ubuntu 16.04, gcc 4.8, python 3.5       Used for openvino-tf CI pre-merge (Python3)

Dockerfile.openvino_tf.build_ovtf_ubuntu1604_py35        Ubuntu 16.04, gcc 4.8, python 3.5       Provided for building with gcc 5+, which provides AVX512 targeting


DEPRECATED (DO NOT USE):

Dockerfile.ov-tf-ci-py2                        Ubuntu 16.04, native gcc, python2        Used for legacy ngraph-tf general python2 builds

Dockerfile.ov-tf-ci-py3                        Ubuntu 16.04, native gcc, python3        Used for legacy ngraph-tf general python3 builds
