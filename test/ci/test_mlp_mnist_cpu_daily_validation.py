# Test intended to be run using pytest
#
# If pytest is not installed on your server, you can install it in a virtual
# environment:
#
#   $ virtualenv -p /usr/bin/python venv-pytest
#   $ source venv-pytest/bin/activate
#   $ pip -U pytest
#   $ pytest test_mnist_cpu_daily_validation.py
#   $ deactivte
#
# This test has no command-line parameters, as it is run via pytest.
# This test does have environment variables that can alter how the run happens:
#
#     Parameter              Purpose & Default (if any)
#
#     TEST_MLP_MNIST_ITERATIONS  Number of iterations (steps) to run;
#                                default=100000
#     TEST_MLP_MNIST_DATA_DIR    Directory where MNIST datafiles are located
#     TEST_MLP_MNIST_LOG_DIR     Optional: directory to write log files to
#
# JUnit XML files can be generated using pytest's command-line options.
# For example:
#
#     $ pytest -s ./test_mnist_cpu_daily_validation.py --junit-xml=../validation_tests_mnist_mlp_cpu.xml --junit-prefix=daily_validation_mnist_mlp_cpu
#

import sys
import os
import re

import lib_validation_testing as VT


# Constants

# Acceptable accuracy
kAcceptableAccuracy = 2.0  # 1.0%, delta must be calculated from percentages

# Default number of iterations (steps) is 100,000
kDefaultSteps = 100000

# As per Yann Lecun's description of the MNIST data-files at URL:
#   http://yann.lecun.com/exdb/mnist/
kTrainEpochSize = 60000  

# Batch size is hard-coded in mnist_softmax_xla.py to 100
kTrainBatchSize = 100

# Relative path (from top of repo) to mnist_softmax_xla.py script
kMnistScriptPath = 'test/mnist_softmax_ngraph.py'

# Python program to run script.  This should just be "python" (or "python2"),
# as the virtualenv relies on PATH resolution to find the python executable
# in the virtual environment's bin directory.
kPythonProg = 'python'


def test_mlp_mnist_cpu_backend():

    # This *must* be run inside the test, because env. var. PYTEST_CURRENT_TEST
    # only exists when inside the test function.
    ngtfDir = VT.findBridgeRepoDirectory()
    script = os.path.join(ngtfDir, kMnistScriptPath)
    VT.checkScript(script)

    dataDir = os.environ.get('TEST_MLP_MNIST_DATA_DIR', None)
    VT.checkMnistData(dataDir)

    iterations = int(os.environ.get('TEST_MLP_MNIST_ITERATIONS', 100000))

    # Run with Google CPU defaults, saving timing and accuracy
    referenceLog = VT.runMlpMnistScript(logID=' Reference',
                                        useNGraph=False,
                                        script=script,
                                        python=kPythonProg,
                                        iterations=iterations,
                                        dataDirectory=dataDir)
    referenceResults = processOutput(referenceLog)

    # Run with NGraph CPU backend, saving timing and accuracy
    VT.checkNGraphEnvironment()
    ngraphLog = VT.runMlpMnistScript(logID=' nGraph',
                                     useNGraph=True,
                                     script=script,
                                     python=kPythonProg,
                                     iterations=iterations,
                                     dataDirectory=dataDir)
    ngraphResults = processOutput(ngraphLog)
    
    lDir = None
    if os.environ.has_key('TEST_MLP_MNIST_LOG_DIR'):
        lDir = os.path.abspath(os.environ['TEST_MLP_MNIST_LOG_DIR'])
        # Dump logs to files, for inclusion in Jenkins artifacts
        VT.writeLogToFile(referenceLog,
                       os.path.join(lDir, 'test_mlp_mnist_cpu_reference.log'))
        VT.writeLogToFile(ngraphLog,
                          os.path.join(lDir, 'test_mlp_mnist_cpu_ngraph.log'))
        # Write Jenkins description, for quick perusal of results
        writeJenkinsDescription(referenceResults, ngraphResults, iterations,
                                os.path.join(lDir,
                                    'test_mlp_mnist_cpu_jenkins_oneline.log'))

    print
    print '----- MNIST-MLP Testing Summary ----------------------------------------'

    summaryLog = None
    if lDir != None:
        summaryLog = os.path.join(lDir, 'test_mlp_mnist_cpu_summary.log')

    logOut = VT.LogAndOutput(logFile=summaryLog)

    # Report commands
    logOut.line()
    logOut.line('Run with default CPU: %s' % referenceResults['command'])
    logOut.line('Run with NGraph CPU: %s' % ngraphResults['command'])

    # Report parameters
    logOut.line()
    logOut.line('Iterations:       %d (aka "steps")' % iterations)
    logOut.line('Batch size:       %d (fixed)' % kTrainBatchSize)
    logOut.line('Epoch size:       %d (fixed)' % kTrainEpochSize)
    logOut.line('nGraph priority:  %d (fixed)' % 70)
    logOut.line('nGraph back-end:  %s (fixed)' % 'CPU')
    logOut.line('Data directory:   %s' % dataDir)

    refAccPercent = float(referenceResults['accuracy']) * 100.0
    ngAccPercent = float(ngraphResults['accuracy']) * 100.0

    # Report accuracy
    deltaAccuracy = abs(refAccPercent - ngAccPercent)
    logOut.line()
    logOut.line('Run with default CPU accuracy: %7.4f%%' % refAccPercent)
    logOut.line('Run with NGraph CPU accuracy: %7.4f%%' % ngAccPercent)
    logOut.line('Accuracy delta: %6.4f%%' % deltaAccuracy)
    logOut.line('Acceptable accuracy delta is <= %6.4f%%'
                % float(kAcceptableAccuracy))
    # Assert for out-of-bounds accuracy
    assert deltaAccuracy <= kAcceptableAccuracy
        
    # Report on times
    logOut.line()
    logOut.line('Run with default CPU took:    %f seconds'
                % referenceResults['wallclock'])
    logOut.line('Run with NGraph CPU took: %f seconds'
                % ngraphResults['wallclock'])
    logOut.line('NGraph was %f times faster than default (wall-clock measurement)'
                % (referenceResults['wallclock'] / ngraphResults['wallclock']))

# End: test_mlp_mnist_cpu_backend()


# Returns array of captured stdout/stderr lines, for post-processing


# Returns dictionary with results extracted from the run:
#     'command':    Command that was run
#     'accuracy':   Accuracy reported for the run
#     'wallclock':  How many seconds the job took to run
def processOutput(log):

    command = None
    accuracy = None
    wallclock = None

    # Dummy processing for proof-of-concept
    lineCount = 0
    for line in log:

        if re.match('Command is:', line):
            if command == None:
                lArray = line.split('"')
                command = str(lArray[1].strip('"'))
                print 'Found command = [%s]' % command
            else:
                raise Exception('Multiple command-is lines found')

        if re.match('Accuracy:', line):
            if accuracy == None:
                lArray = line.split()
                accuracy = float(lArray[1].strip())
                print 'Found accuracy = %f' % accuracy
            else:
                raise Exception('Multiple accuracy lines found')
                
        if re.match('Run length:', line):
            if wallclock == None:
                lArray = line.split()
                wallclock = float(lArray[2].strip())
                print 'Found wallclock = %f' % wallclock
            else:
                raise Exception('Multiple time-elapsed lines found')

        lineCount += 1

    # Make exact zero instead be a very tiny number, to avoid divide-by-zero
    # calculations
    if accuracy == 0.0 or accuracy == None:   accuracy = 0.000000001
    if wallclock == 0.0 or wallclock == None:  wallclock = 0.000000001

    return {'command': command,
            'accuracy': accuracy,
            'wallclock': wallclock}

# End: processOutput


def writeJenkinsDescription(refResults, ngResults, iterations, fileName):

    print 'Jenkins description written to %s' % fileName

    try: 

        fOut = open( fileName, 'w')

        refAccPercent = float(refResults['accuracy']) * 100.0
        ngAccPercent = float(ngResults['accuracy']) * 100.0

        fOut.write( 'MNIST-MLP accuracy - ref: %5.2f%%, ngraph: %5.2f%%, delta %4.2f; ngraph %4.2fx faster; %d steps'
                    % (refAccPercent, ngAccPercent,
                       abs(refAccPercent - ngAccPercent),
                       (refResults['wallclock']/ngResults['wallclock']),
                       iterations))

        fOut.close()

    except Exception as e:
        print 'Unable to write Jenkins description file - %s' % e

# End: writeJenkinsDescription()


