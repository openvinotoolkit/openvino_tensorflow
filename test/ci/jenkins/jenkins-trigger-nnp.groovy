String  PR_URL = CHANGE_URL
String  PR_COMMIT_AUTHOR = CHANGE_AUTHOR
String  PR_TARGET = CHANGE_TARGET
String  JENKINS_BRANCH = "kabhira/update-ng-tf-envvars"
Integer TIMEOUTTIME = "7200"

// Constants
JENKINS_DIR = '.'

timestamps {
    node("trigger") {

        deleteDir()  // Clear the workspace before starting
        // Clone the cje-algo directory which contains our Jenkins groovy scripts
        try {
            sh "git clone -b $JENKINS_BRANCH https://gitlab.devtools.intel.com/AIPG/AlgoVal/cje-algo.git ."
        } catch (e) {
            echo "${e}"
            println("ERROR: An error occurred during cje-algo script checkout.")
            throw e
        }

        echo "Calling ngtf-bridge-ci-premerge.groovy"
        def ngtfbrCIPreMerge = load("${JENKINS_DIR}/ngraph-tf-bridge-ci-premerge.groovy")
        ngtfbrCIPreMerge(PR_URL, PR_COMMIT_AUTHOR, JENKINS_BRANCH, TIMEOUTTIME, PR_TARGET)
        echo "ngtf-bridge-ci-premerge.groovy completed"

    }  // End:  node
}  // End:  timestamps

echo "Done"
