cp ngraph.patch ./tensorflow
cd tensorflow
git apply ngraph.patch
# --test_output=all to see device placement also for passed tests
bazel test  --test_output=all  //tensorflow/python/... 
