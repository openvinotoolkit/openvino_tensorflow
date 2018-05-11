# Test intended to be run using pytest
#
# If pytest is not installed on your server, you can install it in a virtual
# environment:
#
#   $ virtualenv -p /usr/bin/python venv-pytest
#   $ source venv-pytest/bin/activate
#   $ pip -U pytest
#   $ pytest test_convnet_mnist_ngraph_cpu_validation.py
#   $ deactivate
#
# This test has no command-line parameters, as it is run via pytest.
# This test does have environment variables that can alter how the run happens:
#
#     Parameter              Purpose & Default (if any)
#
#     TEST_CONVNET_MNIST_COMPARE_TO  Reference JSON file to compare ngraph
#                                      run to; default = no file
#     TEST_CONVNET_MNIST_ITER        Number of iterations (steps) to run;
#                                      default=1000
#     TEST_CONVNET_MNIST_BATCH_SIZE  Batch size per step; default=128
#
#     TEST_CONVNET_MNIST_DATA_DIR    Directory where MNIST datafiles are located
#     TEST_CONVNET_MNIST_LOG_DIR     Optional: dir to write log files to
#
# JUnit XML files can be generated using pytest's command-line options.
# For example:
# $ pytest -s ./test_convnet_mnist_cpu_daily_validation.py --junit-xml=../validation_tests_convnet_mnist_cpu.xml --junit-prefix=daily_validation_convnet_mnist_cpu
#
# Example of ngraph command-line used:
# $ TODO
#

import sys
import os
import re
import json
import datetime as DT

import lib_validation_testing as VT


# Constants

# Log files
kConvnetCPUNgLog          = 'test_convnet_mnist_cpu_reference.log'
kConvnetCPUNgJson         = 'test_convnet_mnist_cpu_reference.json'
kConvnetRefSummaryLog     = 'test_convnet_mnist_cpu_reference_summary.log'
kConvnetJenkinsRefSummLog = 'test_convnet_mnist_cpu_jenkins_ref_oneline.log'

# Acceptable accuracy
kAcceptableAccuracyDelta = 0.02  # 0.02 = 2%

# Default number of iterations (steps) is 1000
kDefaultIter = 1000

# As per Yann Lecun's description of the MNIST data-files at URL:
#   http://yann.lecun.com/exdb/mnist/
kTrainEpochSize = 60000  

# For now, batch size is a constant
kTrainBatchSize = 128

# Relative path (from top of bridge repo) to cifar10_main.py script
kConvnetScriptPath = 'test/mnist_deep_simplified.py'

# Python program to run script.  This should just be "python" (or "python2"),
# as the virtualenv relies on PATH resolution to find the python executable
# in the virtual environment's bin directory.
kPythonProg = 'python'

# Date and time that testing started, for the run stats collected later
kRunDateTime = DT.datetime.now()


def test_convnet_mnist_ngraph_cpu_backend():

    # This *must* be run inside the test, because env. var. PYTEST_CURRENT_TEST
    # only exists when inside the test function.
    ngtfDir = VT.findBridgeRepoDirectory()
    script = os.path.join(ngtfDir, kConvnetScriptPath)
    VT.checkScript(script)

    iterations = int(os.environ.get('TEST_CONVNET_MNIST_ITER', kDefaultIter))

    dataDir = os.environ.get('TEST_CONVNET_MNIST_DATA_DIR', None)
    VT.checkMnistData(dataDir)

    # Run with NGraph CPU backend, saving timing and accuracy
    refLog = VT.runConvnetScript(logID=' Reference',
                                 useNGraph=False,
                                 script=script,
                                 python=kPythonProg,
                                 iterations=iterations,
                                 batchSize=kTrainBatchSize,
                                 dataDirectory=dataDir,
                                 verbose=False)  # log-device-placement
    refResults = \
        VT.collect_convnet_mnist_results(runType='Reference CPU backend',
                                         log=refLog,
                                         date=str(kRunDateTime),
                                         iterations=iterations,
                                         batchSize=kTrainBatchSize)

    print
    print('Collected results for *reference* in this run are:')
    print(json.dumps(refResults, indent=4))
    
    lDir = None
    if os.environ.has_key('TEST_CONVNET_MNIST_LOG_DIR'):
        lDir = os.path.abspath(os.environ['TEST_CONVNET_MNIST_LOG_DIR'])
        # Dump logs to files, for inclusion in Jenkins artifacts
        VT.writeLogToFile(refLog,
                          os.path.join(lDir, kConvnetCPUNgLog))
        VT.writeJsonToFile(json.dumps(refResults, indent=4),
                           os.path.join(lDir, kConvnetCPUNgJson))

    print
    print '----- CONVNET MNIST Reference Testing Summary ----------------------------------------'

    summaryLog = None
    if lDir != None:
        summaryLog = os.path.join(lDir, kConvnetRefSummaryLog)

    logOut = VT.LogAndOutput(logFile=summaryLog)

    # Report commands
    logOut.line()
    logOut.line('Reference run with CPU: %s' % refResults['command'])

    # Report parameters
    logOut.line()
    logOut.line('Iterationss:           %d' % iterations)
    logOut.line('Batch size:            %d' % kTrainBatchSize)
    logOut.line('Epoch size:            %d (fixed in MNIST)'
                % kTrainEpochSize)
    logOut.line('nGraph back-end used:  %s' % 'CPU')
    logOut.line('Data directory:        %s' % dataDir)

    refAccuracy = float(refResults['accuracy'])

    # Report accuracy
    logOut.line()
    logOut.line('Accuracy, in reference run using CPU: %7.6f' % refAccuracy)

    # Report on times
    logOut.line()
    logOut.line('Reference run using CPU took: %f seconds'
                % refResults['wallclock'])

    # Make sure all output has been flushed before running assertions
    logOut.flush()

# End: test_convnet_mnist_cpu_backend()
