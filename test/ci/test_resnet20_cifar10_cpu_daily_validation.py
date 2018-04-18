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
#     TEST_RESNET20_CIFAR10_EPOCHS     Number of epochs (steps) to run;
#                                      default=250
#     TEST_RESNET20_CIFAR10_DATA_DIR   Directory where CIFAR10 datafiles are
#                                      located
#     TEST_RESNET20_CIFAR10_LOG_DIR    Optional: directory to write log files to
#
# JUnit XML files can be generated using pytest's command-line options.
# For example:
# $ pytest -s ./test_resnet20_cifar10_cpu_daily_validation.py --junit-xml=../validation_tests_resnet20_cifar10_cpu.xml --junit-prefix=daily_validation_resnet20_cifar10_cpu
#
# Example of ngraph command-line used:
# $ KMP_BLOCKTIME=1 OMP_NUM_THREADS=56 KMP_AFFINITY=granularity=fine,compact,1,0 python cifar10_main.py --data_dir /tmp/cifar10_input_data --train_epochs 1 --batch_size 128 --resnet_size=20 --select_device NGRAPH --model_dir /tmp/cifar10-ngraph --epochs_between_evals 1 --inter_op 2
#

import sys
import os
import re
import json
import datetime as DT

import lib_validation_testing as VT


# Constants

# Log files
kResnet20CPURefLog         = 'test_resnet20_cifar10_cpu_reference.log'
kResnet20CPUNgLog          = 'test_resnet20_cifar10_cpu_ngraph.log'
kResnet20SummaryLog        = 'test_resnet20_cifar10_cpu_summary.log'
kResnet20JenkinsSummaryLog = 'test_resnet20_cifar10_cpu_jenkins_oneline.log'
kResnet20CPURefJson        = 'test_resnet20_cifar10_cpu_reference.json'
kResnet20CPUNgJson         = 'test_resnet20_cifar10_cpu_ngraph.json'

# Acceptable accuracy
kAcceptableAccuracyDelta = 0.02  # 0.02 = 2%

# Default number of epochs (steps) is 250
kDefaultEpochs = 250

# As per Alex Krizhevsky's description of the CIFAR-10 dataset at:
#   https://www.cs.toronto.edu/~kriz/cifar.html
kTrainEpochSize = 50000  

# Batch size is set with --data_dir.  For now, it is a constant
kTrainBatchSize = 128

# Relative path (from top of bridge repo) to cifar10_main.py script
kResnet20ScriptPath = 'test/resnet/cifar10_main.py'

# Python program to run script.  This should just be "python" (or "python2"),
# as the virtualenv relies on PATH resolution to find the python executable
# in the virtual environment's bin directory.
kPythonProg = 'python'

# Date and time that testing started, for the run stats collected later
kRunDateTime = DT.datetime.now()


def test_resnet20_cifar10_cpu_backend():

    # This *must* be run inside the test, because env. var. PYTEST_CURRENT_TEST
    # only exists when inside the test function.
    ngtfDir = VT.findBridgeRepoDirectory()
    script = os.path.join(ngtfDir, kResnet20ScriptPath)
    VT.checkScript(script)

    dataDir = os.environ.get('TEST_RESNET20_CIFAR10_DATA_DIR', None)
    if not os.path.isdir(dataDir):
        raise Exception('Directory %s does not exist' % str(dataDir))

    epochs = int(os.environ.get('TEST_RESNET20_CIFAR10_EPOCHS', kDefaultEpochs))

    # Run with Google CPU defaults, saving timing and accuracy
    referenceLog = VT.runResnetScript(logID=' Reference',
                                      useNGraph=False,
                                      script=script,
                                      python=kPythonProg,
                                      epochs=epochs,
                                      batchSize=kTrainBatchSize,
                                      dataDirectory=dataDir,
                                      verbose=False)  # log-device-placement
    referenceResults = collect_results(runType='TensorFlow Default CPU',
                                       log=referenceLog, date=str(kRunDateTime),
                                       epochs=epochs, batchSize=kTrainBatchSize)
    print
    print('Collected results for *reference*:')
    print(json.dumps(referenceResults, indent=4))

    # Run with NGraph CPU backend, saving timing and accuracy
    VT.checkNGraphEnvironment()
    ngraphLog = VT.runResnetScript(logID=' nGraph',
                                   useNGraph=True,
                                   script=script,
                                   python=kPythonProg,
                                   epochs=epochs,
                                   batchSize=kTrainBatchSize,
                                   dataDirectory=dataDir,
                                   verbose=False)  # log-device-placement
    ngraphResults = collect_results(runType='nGraph CPU backend',
                                    log=ngraphLog, date=str(kRunDateTime),
                                    epochs=epochs, batchSize=kTrainBatchSize)
    print
    print('Collected results for *nGraph*:')
    print(json.dumps(ngraphResults, indent=4))
    
    lDir = None
    if os.environ.has_key('TEST_RESNET20_CIFAR10_LOG_DIR'):
        lDir = os.path.abspath(os.environ['TEST_RESNET20_CIFAR10_LOG_DIR'])
        # Dump logs to files, for inclusion in Jenkins artifacts
        VT.writeLogToFile(referenceLog,
                          os.path.join(lDir, kResnet20CPURefLog))
        VT.writeLogToFile(ngraphLog,
                          os.path.join(lDir, kResnet20CPUNgLog))
        VT.writeJsonToFile(json.dumps(referenceResults, indent=4),
                           os.path.join(lDir, kResnet20CPURefJson))
        VT.writeJsonToFile(json.dumps(ngraphResults, indent=4),
                           os.path.join(lDir, kResnet20CPUNgJson))
        # Write Jenkins description, for quick perusal of results
        write_jenkins_description(referenceResults, ngraphResults, epochs,
                                  os.path.join(lDir,
                                               kResnet20JenkinsSummaryLog))

    print
    print '----- RESNET20 CIFAR10 Testing Summary ----------------------------------------'

    summaryLog = None
    if lDir != None:
        summaryLog = os.path.join(lDir, kResnet20SummaryLog)

    logOut = VT.LogAndOutput(logFile=summaryLog)

    # Report commands
    logOut.line()
    logOut.line('Run with default CPU: %s' % referenceResults['command'])
    logOut.line('Run with NGraph CPU: %s' % ngraphResults['command'])

    # Report parameters
    logOut.line()
    logOut.line('Epochs:                %d' % epochs)
    logOut.line('Batch size:            %d' % kTrainBatchSize)
    logOut.line('Epoch size:            %d (fixed in CIFAR10)'
                % kTrainEpochSize)
    logOut.line('nGraph back-end used:  %s' % 'CPU')
    logOut.line('Data directory:        %s' % dataDir)

    refAccuracy = float(referenceResults['accuracy'])
    ngAccuracy = float(ngraphResults['accuracy'])

    # Report accuracy
    acceptableDelta = refAccuracy * kAcceptableAccuracyDelta
    deltaAccuracy = abs(refAccuracy - ngAccuracy)
    logOut.line()
    logOut.line('Accuracy, in run with default CPU:             %7.6f' % refAccuracy)
    logOut.line('Accuracy, in run with NGraph CPU:          %7.6f' % ngAccuracy)
    logOut.line('Acceptable accuracy range (from reference) is: %4.2f%% of %7.6f'
                % (kAcceptableAccuracyDelta * 100, refAccuracy))
    logOut.line('Acceptable accuracy delta is <= %7.6f'
                % float(acceptableDelta))
    logOut.line('Actual accuracy delta is %7.6f' % deltaAccuracy)
    # Report on times
    logOut.line()
    logOut.line('Run with default CPU took:    %f seconds'
                % referenceResults['wallclock'])
    logOut.line('Run with NGraph CPU took: %f seconds'
                % ngraphResults['wallclock'])
    logOut.line('NGraph was %f times longer than default (wall-clock measurement)'
                % (ngraphResults['wallclock'] / referenceResults['wallclock']))

    # Make sure all output has been flushed before running assertions
    logOut.flush()

    # All assertions are now done at the very end of the run, after all of
    # the summary output has been written.

    assert deltaAccuracy <= acceptableDelta  # Assert for out-of-bounds accuracy
        
# End: test_resnet20_cifar10_cpu_backend()


# Returns dictionary with results extracted from the run:
#     'run-type':       Short description of run being done
#     'command':        Command that was run
#     'date-time':      Date and time (on server) when command was run
#     'server':         Name of server on which command was run
#     'cpu-model-name': Model-name of CPUs on the server
#     'epochs':         Epochs run
#     'batch-size':     Batch size for run
#     'accuracy':       Accuracy reported for the run
#     'loss':           Loss reported at the end of the run
#     'wallclock':      How many seconds the job took to run
def collect_results(runType=None, date=None, epochs=None, batchSize=None,
                    log=None):

    runType = runType        # Passed in as a paramter
    command = None           # Derived from log
    date_time = date         # Passed in as a parameter
    server = None            # Derived from process environment
    cpu_model = None         # Derived from /proc/cpuinfo, if present
    epochs = epochs          # Passed in as a parameter
    batchSize = batchSize    # Passed in as a parameter
    accuracy = None          # Derived from log
    loss = None              # Derived from log
    wallclock = None         # Derived from log

    for line in log:  # Read through log

        if re.match('Command is:', line):
            if command == None:
                lArray = line.split('"')
                command = str(lArray[1].strip('"'))
                print 'Found command = [%s]' % command
            else:
                raise Exception('Multiple command-is lines found')

        if re.search("'accuracy':", line):
            if accuracy == None:
                print('Found accuracy: "%s"' % line.strip())
                lArray = line.split()
                accuracy = float(lArray[5].strip().rstrip('}'))
                print('Computed accuracy = %f' % accuracy)
            else:
                raise Exception('Multiple accuracy lines found')
                
        if re.match("{'loss':", line):
            if loss == None:
                print('Found loss: "%s"' % line.strip())
                lArray = line.split()
                loss = float(lArray[1].strip().rstrip(','))
                print('Computed loss = %f' % loss)
            else:
                raise Exception('Multiple loss lines found')
                
        if re.match('^Run length:', line):
            if wallclock == None:
                print('Found run-length: "%s"' % line.strip())
                lArray = line.split()
                wallclock = float(lArray[2].strip())
                print('Computed wallclock = %f' % wallclock)
            else:
                raise Exception('Multiple time-elapsed lines found')

    # Derive the server used.  This is a little tricky, as often this script
    # is run in a Docker container.  Therefore, looked for a passed in
    # environment variable (HOST_HOSTNAME) first, followed by HOSTNAME
    if os.environ.has_key('HOST_HOSTNAME'):
        server = '%s-docker' % os.environ['HOST_HOSTNAME']
    elif os.environ.has_key('HOSTNAME'):
        server = os.environ['HOSTNAME']
   

    # Derive the CPU model-name from /proc/cpuinfo
    if os.path.isfile('/proc/cpuinfo'):
        fIn = open('/proc/cpuinfo', 'r')
        lines = filter(lambda x: re.match('model name', x), fIn.readlines())
        fIn.close()
        if len(lines) > 0:
            # Strip the first line of edge-whitespace, then break into fields
            fields = (lines[0].strip()).split(':')
            # The model name is field 1 (with key "model name" as field 0)
            cpu_model = fields[1].strip()

    # Make exact zero instead be a very tiny number, to avoid divide-by-zero
    # calculations
    if accuracy == 0.0 or accuracy == None:  accuracy = 0.000000001
    if wallclock == 0.0 or wallclock == None:  wallclock = 0.000000001

    return {'run-type': runType,
            'command': command,
            'date-time': date_time,
            'server': server,
            'cpu-model-name': cpu_model,
            'epochs': epochs,
            'batch-size': batchSize,
            'accuracy': accuracy,
            'loss': loss,
            'wallclock': wallclock}

# End: collect_results


def write_jenkins_description(refResults, ngResults, epochs, fileName):

    print 'Jenkins description written to %s' % fileName

    try: 

        fOut = open( fileName, 'w')

        refAccuracy = float(refResults['accuracy'])
        ngAccuracy = float(ngResults['accuracy'])

        acceptableDelta = refAccuracy * kAcceptableAccuracyDelta
        deltaAccuracy = abs(refAccuracy - ngAccuracy)

        fOut.write( 'Resnet-CIFAR10 accuracy - ref: %5.4f, ngraph: %5.4f, delta %5.4f; ngraph %4.2fx slower; %d epochs'
                    % (refAccuracy, ngAccuracy, deltaAccuracy,
                       (ngResults['wallclock']/refResults['wallclock']),
                       epochs))

        fOut.close()

    except Exception as e:
        print 'Unable to write Jenkins description file - %s' % e

# End: write_jenkins_description()
