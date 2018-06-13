# OS X HOWTO for `ngraph-tf` and dependencies

## Installation

1. Install [bazel](https://github.com/bazelbuild/bazel/releases).  [0.11.1 works](https://github.com/bazelbuild/bazel/releases/tag/0.11.1), or, if you're feeling adventurous, you could try a later version.
2. `port install coreutils`, then add `/opt/local/libexec/gnubin` **in front** of your `$PATH`.  Both `tensorflow` and `ngraph` assume GNU userland tools, and you'll run into errors otherwise.
3. Make and activate yourself a virtualenv that we'll be using for our custom-built TensorFlow.
4. Build TensorFlow and its framework for unit tests:

	```
	git clone git@github.com:tensorflow/tensorflow.git
	pushd tensorflow
	git checkout r1.8
	./configure # you can disable everything here if you like, or stick with defaults
	bazel run //tensorflow/tools/pip_package:build_pip_package /tmp/tensorflow_pkg
	pip install /tmp/tensorflow_pkg/tensorflow*.whl
	bazel build --config=opt //tensorflow:libtensorflow_cc.so
	popd
	```

5. Prepare `ngraph-tf` for the build:

	```
	git clone git@github.com:NervanaSystems/ngraph-tf.git
	pushd ngraph-tf.git
	git checkout igor/osx
	ln -s ../tensorflow
	mkdir build && cd build
	cmake -DNGRAPH_USE_PREBUILT_LLVM=False -DCMAKE_BUILD_TYPE=Debug ..
	```

6. `vim CMakeFiles/ext_ngraph.dir/build.make` and replace `$(nproc)` with something more reasonable, like 4 or 6, otherwise `Make` will spawn unlimited processes and lock up your machine.  (*TODO*: figure out why that happens even when `nproc` exists and should return 8.)
7. `make -j 4` (or maybe 6)
8. `cd test && make -j 4`
9. Add `<path-to-tensorflow-repo>/bazel-out/darwin-py3-opt/bin/tensorflow` and `<path-to-ngraph-tf-repo>/build/ngraph/ngraph_dist/lib` to your `$LD_LIBRARY_PATH` and `$DYLD_LIBRARY_PATH`
10. `./gtest_ngtf`

## Debugging

Don't just use `lldb` -- it likely refers to `/usr/bin/lldb` and OS X security preferences will prevent it from inheriting your `$LD_LIBRARY_PATH`.  Instead, alias it to `/Applications/Xcode.app/Contents/Developer/usr/bin/lldb`.
