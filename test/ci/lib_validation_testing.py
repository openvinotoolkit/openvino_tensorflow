# Library of common classes and functions for validation tests


import sys
import os
import re
import subprocess
import time
import datetime as DT
from random import randint


# =============================================================================
# Common Functions
# =============================================================================

def findBridgeRepoDirectory():

    # Bridge directory is parent of dir where the validation test resides

    if not os.environ['PYTEST_CURRENT_TEST']:
        raise Exception('Test does not have pytest-provided PYTEST_CURRENT_TEST environment variable')

    pytestCurrent = os.environ['PYTEST_CURRENT_TEST'].split(':')
    testFile = pytestCurrent[0]

    # scriptDir = os.path.abspath(os.path.dirname(testFile))
    bridgeDir = os.path.abspath(os.path.join(os.path.dirname(testFile), '..', '..'))

    return bridgeDir

# End: findBridgeRepoDirectory()


def checkNGraphEnvironment():

    if not os.environ.has_key('LD_LIBRARY_PATH'):
        raise Exception('LD_LIBRARY_PATH environment variable has not been set')
    
    else:
        if not os.path.isfile(os.path.join(os.environ['LD_LIBRARY_PATH'],
                                           'libngraph.so')):
            raise Exception('Could not find libngraph.so in LD_LIBRARY_PATH')

# End: def checkNGraphEnvironment()


def checkScript(scriptFile):

    if not os.path.isfile(scriptFile):
        raise Exception('WARNING: Script path is not a file: %s'
                        % str(scriptFile))

# End: def checkScript()


def timeElapsedString(startTime, endTime):

    timeElapsed = endTime - startTime

    return('Run length: %s seconds (%s)'
           % (timeElapsed.total_seconds(), str(timeElapsed)))

# End: def timeElapsedString()


def writeLogToFile(logArray, fileName):

    print 'Log written to %s' % fileName

    fOut = open(fileName, 'w')
    for line in logArray:  fOut.write('%s\n' % str(line))
    fOut.close()

# End: writeLogToFile()


def writeJsonToFile(jsonString, fileName):

    print 'JSON results written to %s' % fileName

    fOut = open(fileName, 'w')
    fOut.write('%s\n' % str(jsonString))
    fOut.close()


class LogAndOutput(object) :


    def __init__(self, logFile=None):

        if logFile == None :
            self.out = None

        else:
            try:
                self.out = open(logFile, 'w')
            except Exception as e:
                raise Exception('Unable to open log-file %s in LogAndOutput() due to exception: %s'
                                % (logFile, e))
        # End: else

    # End: def __init__()


    def  line(self, message=''):

        print('%s' % str(message))
        if self.out != None: self.out.write('%s\n' % str(message))

    # End: def print()

    def  flush( self ):

        sys.stdout.flush()
        if self.out != None: self.out.flush()

# End: LogAndOutput()




def runCommand(command=None,  # Script to run
               logID=''):     # Log line prefix

    print
    print 'Command being run with:'
    print '    command: %s' % str(command)
    print '    logID:   %s' % str(logID)

    quiet = False   # Used for filtering messages, not currently activated
    patterns = [ ]

    log = []

    if command == None:
        raise Exception('runCommand() called with empty command parameter')

    cmd = command

    cmdMsg = 'Command is: "%s"' % str(cmd)
    print cmdMsg
    log.append(cmdMsg)

    sTime = DT.datetime.today()
    subP = subprocess.Popen(cmd,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    print 'Subprocess started at %s' % str(sTime)
    print 'Subprocess started with PID %d' % subP.pid

    retCode = None
    while True:
        retCode = subP.poll()
        if not retCode == None: break
        line = subP.stdout.readline()
        if line != None:
            log.append(line.strip())
            timeStr = str(DT.datetime.time(DT.datetime.now()))
            if not quiet:
                sys.stdout.write('%s%s: %s\n' % (timeStr, logID, line.strip()))
            else:
                if any(re.match(regex, line) for regex in patterns):
                    sys.stdout.write('%s%s: %s\n'
                                     % (timeStr, logID, line.strip()))
            sys.stdout.flush()

    eTime = DT.datetime.now()
    print 'Subprocess completed at %s' % str(eTime)
    elapsed = timeElapsedString(sTime,eTime)
    log.append(elapsed)
    print(elapsed)

    subP = None  # Release the subprocess Popen object

    if retCode != 0:
        print('ERROR: Subprocess (%s) returned non-zero exit code %d'
              % (cmd, retCode))
    else:
        print('Subprocess returned exit code %d' % retCode)

    assert retCode == 0  # Trigger a formal assertion

    return log

# End: def runCommand()


# =============================================================================
# MNIST-Specific Functions
# =============================================================================


def  checkMnistData(dataDir):

    dataFiles = ['t10k-images-idx3-ubyte.gz',
                 't10k-labels-idx1-ubyte.gz',
                 'train-images-idx3-ubyte.gz',
                 'train-labels-idx1-ubyte.gz']

    if dataDir == None:
        raise Exception('Data directory was not specified in TEST_*_MNIST_DATA_DIR')

    if not os.path.isdir(dataDir):
        raise Exception('Data directory %s is not actually a directory'
                        % str(dataDir))

    for f in dataFiles:
        if not os.path.isfile(os.path.join(dataDir, f)):
            raise Exception('Data file %s not found in %s'
                            % (f, dataDir))

# End: checkMnistData()                          


# =============================================================================
# MLP-Specific Functions
# =============================================================================


def runMlpMnistScript(script=None,          # Script to run
                      useNGraph=True,       # False->reference, True->nGraph++
                      dataDirectory=None,   # --data_dir, where MNIST data is
                      iterations=None,      # Iterations is the same as steps
                      python=None,          # Which python to use
                      logID=''):            # Log line prefix

    print
    print 'MNIST script being run with:'
    print '    script:         %s' % str(script)
    print '    useNGraph:      %s' % str(useNGraph)
    print '    dataDirectory:  %s' % str(dataDirectory)
    print '    iterations:     %s' % str(iterations)
    print '    python:         %s' % str(python)
    print '    logID:          %s' % str(logID)

    if dataDirectory != None:
        optDataDir = '--data_dir %s' % dataDirectory
        print 'Using data directory %s' % dataDirectory
    else: optDataDir = ''

    # -u puts python in unbuffered mode
    cmd = ('%s -u %s %s --train_loop_count %d'
           % (python, script, optDataDir, iterations))

    if useNGraph:
        print 'Setting up run in nGraph environment'
        cmd = 'OMP_NUM_THREADS=56 KMP_AFFINITY=granularity=fine,scatter %s' % cmd
        cmd = '%s --select_device NGRAPH' % cmd
    else:
        print 'Setting up run in reference CPU environment (no nGraph)'
        cmd = '%s --select_device CPU' % cmd

    # Hook for testing results detection without having to run multi-hour
    # FW+Dataset tests
    if (os.environ.has_key('TF_NG_DO_NOT_RUN')
        and len(os.environ['TF_NG_DO_NOT_RUN']) > 0):
        runLog = runFakeCommand(command=cmd, logID=logID)
    else:
        runLog = runCommand(command=cmd, logID=logID)

    return runLog

# End: def runMnistScript()


# =============================================================================
# Convnet-Specific Functions
# =============================================================================


def runConvnetScript(script=None,        # Script to run
                     useNGraph=True,     # False->reference, True->nGraph++
                     dataDirectory=None, # --data_dir, where MNIST data is
                     iterations=None,    # Epochs to run
                     batchSize=128,      # Batch size for training
                     python=None,        # Which python to use
                     verbose=False,      # If True, enable log_device_placement
                     logID=''):          # Log line prefix

    print
    print 'Convnet script being run with:'
    print '    script:         %s' % str(script)
    print '    useNGraph:      %s' % str(useNGraph)
    print '    dataDirectory:  %s' % str(dataDirectory)
    print '    iterations:     %s' % str(iterations)
    print '    python:         %s' % str(python)
    print '    logID:          %s' % str(logID)


    if iterations is None:
        raise Exception('runConvnetScript() called without parameter iterations')

    if batchSize is None:
        raise Exception('runConvnetScript() called without parameter batchSize')

    # -u puts python in unbuffered mode
    cmd = ('%s -u %s --data_dir %s --train_loop_count %d --batch_size %d'
           % (python, script, dataDirectory, int(iterations), int(batchSize)))

    if useNGraph:
        print 'Setting up run in nGraph environment'
        cmd = 'OMP_NUM_THREADS=44 KMP_BLOCKTIME=0 KMP_AFFINITY=granularity=fine,compact,1,0 %s' % cmd
    else:
        print 'Setting up run in reference CPU environment (no nGraph)'
        # Now that we use MKLDNN for reference runs, we need to make sure it
        # runs using appropriate OMP and KMP settings
        cmd = 'OMP_NUM_THREADS=44 KMP_BLOCKTIME=0 KMP_AFFINITY=granularity=fine,compact,1,0 %s' % cmd
        cmd = '%s --use_xla_cpu 1' % cmd

    if verbose:
        print 'Enabling verbose output (--log_device_placement)'
        cmd = '%s --log_device_placement True' % cmd

    # Hook for testing results detection without having to run multi-hour
    # Framework+Dataset tests
    if (os.environ.has_key('TF_NG_DO_NOT_RUN')
        and len(os.environ['TF_NG_DO_NOT_RUN']) > 0):
        runLog = runFakeCommand(command=cmd, logID=logID)
    else:
        runLog = runCommand(command=cmd, logID=logID)

    return runLog

# End: def runConvnetScript()


# Returns dictionary with results extracted from the run:
#     'run-type':       Short description of run being done
#     'command':        Command that was run
#     'date-time':      Date and time (on server) when command was run
#     'server':         Name of server on which command was run
#     'cpu-model-name': Model-name of CPUs on the server
#     'iterations':     Iterationss run
#     'batch-size':     Batch size for run
#     'accuracy':       Accuracy reported for the run
#     'loss':           Loss reported at the end of the run
#     'wallclock':      How many seconds the job took to run
def collect_convnet_mnist_results(runType=None, date=None,
                                  iterations=None, batchSize=None,
                                  log=None):

    runType = runType        # Passed in as a paramter
    command = None           # Derived from log
    date_time = date         # Passed in as a parameter
    server = None            # Derived from process environment
    cpu_model = None         # Derived from /proc/cpuinfo, if present
    iterations = iterations  # Passed in as a parameter
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

        if re.search("test accuracy", line):
            if accuracy == None:
                print('Found accuracy: "%s"' % line.strip())
                lArray = line.split()
                accuracy = float(lArray[2].strip())
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
            'iterations': iterations,
            'batch-size': batchSize,
            'accuracy': accuracy,
            'loss': loss,
            'wallclock': wallclock}

# End: collect_convnet_mnist_results


def write_jenkins_convnet_mnist_description(refResults, ngResults,
                                            acceptableDelta,
                                            iterations, fileName):

    print 'Jenkins description written to %s' % fileName

    try: 

        fOut = open( fileName, 'w')

        refAccuracy = float(refResults['accuracy'])
        ngAccuracy = float(ngResults['accuracy'])

        acceptableDelta = refAccuracy * acceptableDelta
        deltaAccuracy = abs(refAccuracy - ngAccuracy)

        fOut.write( 'Convnet-MNIST accuracy - ref: %5.4f, ngraph: %5.4f, delta %5.4f; ngraph %4.2fx faster; %d iterations'
                    % (refAccuracy, ngAccuracy, deltaAccuracy,
                       (refResults['wallclock']/ngResults['wallclock']),
                       iterations))

        fOut.close()

    except Exception as e:
        print 'Unable to write Jenkins description file - %s' % e

# End: write_jenkins_convnet_mnist_description()


# =============================================================================
# Resnet-Specific Functions
# =============================================================================


def runResnetScript(script=None,          # Script to run
                    useNGraph=True,       # False->reference, True->nGraph++
                    dataDirectory=None,   # --data_dir, where MNIST data is
                    epochs=None,          # Epochs to run
                    epochsPerEval=None,   # Epochs per eval cycle
                    batchSize=128,        # Batch size for training
                    python=None,          # Which python to use
                    verbose=False,        # If True, enable log_device_placement
                    logID=''):            # Log line prefix

    print
    print 'Resnet20 script being run with:'
    print '    script:         %s' % str(script)
    print '    useNGraph:      %s' % str(useNGraph)
    print '    dataDirectory:  %s' % str(dataDirectory)
    print '    epochs:         %s' % str(epochs)
    print '    epochsPerEval:  %s' % str(epochsPerEval)
    print '    python:         %s' % str(python)
    print '    logID:          %s' % str(logID)


    if epochs is None:
        raise Exception('runResnetScript() called without parameter epochs')

    if batchSize is None:
        raise Exception('runResnetScript() called without parameter batchSize')

    # -u puts python in unbuffered mode
    cmd = ('%s -u %s --data_dir %s --train_epochs %d --batch_size %d --resnet_size=20'
           % (python, script, dataDirectory, int(epochs), int(batchSize)))

    if useNGraph:
        print 'Setting up run in nGraph environment'
        cmd = 'OMP_NUM_THREADS=44 KMP_BLOCKTIME=1 KMP_AFFINITY=granularity=fine,compact,1,0 %s' % cmd
        cmd = '%s --select_device NGRAPH' % cmd
        cmd = '%s --model_dir /tmp/cifar10-ngraph' % cmd
        cmd = '%s --inter_op 2' % cmd
        cmd = '%s --data_format channels_first' % cmd
    else:
        print 'Setting up run in reference CPU environment (no nGraph)'
        # Now that we use MKLDNN for reference runs, we need to make sure it
        # runs using appropriate OMP and KMP settings
        cmd = 'OMP_NUM_THREADS=44 KMP_BLOCKTIME=1 KMP_AFFINITY=granularity=fine,compact,1,0 %s' % cmd
        cmd = '%s --select_device CPU' % cmd
        cmd = '%s --model_dir /tmp/cifar10-reference' % cmd

    # If --epochs_per_eval is greater than --train_epochs, then no epochs
    # will actually run, due to a limitation in the script.  Fix this.
    if (epochsPerEval is None) or (int(epochsPerEval) > int(epochs)):
        epochsPerEval = int(epochs)

    if epochsPerEval is not None:
        print 'Adding --epochs_between_evals %d' % int(epochsPerEval)
        cmd = '%s --epochs_between_evals %d' % (cmd, int(epochsPerEval))

    if verbose:
        print 'Enabling verbose output (--log_device_placement)'
        cmd = '%s --log_device_placement True' % cmd

    # Hook for testing results detection without having to run multi-hour
    # Framework+Dataset tests
    if (os.environ.has_key('TF_NG_DO_NOT_RUN')
        and len(os.environ['TF_NG_DO_NOT_RUN']) > 0):
        runLog = runFakeCommand(command=cmd, logID=logID)
    else:
        runLog = runCommand(command=cmd, logID=logID)

    return runLog

# End: def runResnetScript()


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
def collect_resnet20_cifar10_results(runType=None, date=None,
                                     epochs=None, batchSize=None,
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

# End: collect_resnet20_cifar10_results


def write_jenkins_resnet20_cifar10_description(refResults, ngResults,
                                               acceptableDelta,
                                               epochs, fileName):

    print 'Jenkins description written to %s' % fileName

    try: 

        fOut = open( fileName, 'w')

        refAccuracy = float(refResults['accuracy'])
        ngAccuracy = float(ngResults['accuracy'])

        acceptableDelta = refAccuracy * acceptableDelta
        deltaAccuracy = abs(refAccuracy - ngAccuracy)

        fOut.write( 'Resnet-CIFAR10 accuracy - ref: %5.4f, ngraph: %5.4f, delta %5.4f; ngraph %4.2fx faster; %d epochs'
                    % (refAccuracy, ngAccuracy, deltaAccuracy,
                       (refResults['wallclock']/ngResults['wallclock']),
                       epochs))

        fOut.close()

    except Exception as e:
        print 'Unable to write Jenkins description file - %s' % e

# End: write_jenkins_description()


def runFakeCommand(command=None, logID=''):

    print('')
    print('Fake command being run, to test testing infrastructure:')
    print('    logID: %s' % str(logID))
    print('')
    print('Would have run:')
    print('    command: %s' % str(command))

    sTime = DT.datetime.today()
    print('Fake command started at %s' % str(sTime))

    # Sleep a random amount, so we can test 
    time.sleep(randint(5,15))

    eTime = DT.datetime.now()
    print('Fake command completed at %s' % str(eTime))
    elapsed = timeElapsedString(sTime, eTime)
    print(elapsed)

    return(['Fake log',
            'Nothing run',
            "{'loss': 15.762859, 'global_step': 391, 'accuracy': 0.1045}",
            elapsed])


# End: runFakeCommand()
