#!  /bin/bash

# This script is a wrapper which captures the output of a script and writes
# a proper JUnit XML file, for consumption by Jenkins.  This was adapated
# from run-integration-tests.sh at path
# tensorflow/compiler/plugin/ngraph/tests/integration-tests in repo
# NervanaSystems/ngraph-tensorflow-1.3

# Before calling this script, you should set three environment variables:
#
#   JUNIT_WRAP_FILE   Name of the file and path to write XML output to
#   JUNIT_WRAP_SUITE  Name of the JUnit "suite" to record results for
#   JUNIT_WRAP_TEST   Name of the JUnit "test" to record results for
#
# Note that only one test is run in the suite, with the test having a
# pass/fail status based on the exit code of the command

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------

declare JUNIT_XML=''  # XML for testsuites and testcases
declare -i JUNIT_SUITES=0
declare -i JUNIT_TESTS=0
declare -i JUNIT_PASS=0
declare -i JUNIT_FAIL=0
declare -i JUNIT_DISABLED=0
declare JUNIT_SUITE_NAME='UNDEFINED_SUITE'
declare JUNIT_SUITE_XML=''  # XML for testcases
declare -i JUNIT_SUITE_TESTS=0
declare -i JUNIT_SUITE_PASS=0
declare -i JUNIT_SUITE_FAIL=0
declare -i JUNIT_SUITE_DISABLED=0
declare NL=$'\n'  # Used to insert newlines in double-quoted strings

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

xml_start_suite() {

    JUNIT_SUITE_NAME="${JUNIT_WRAP_SUITE}"
    JUNIT_SUITE_TESTS=0
    JUNIT_SUITE_PASS=0
    JUNIT_SUITE_FAIL=0
    JUNIT_SUITE_DISABLED=0

    JUNIT_SUITE_XML=''
    (( JUNIT_SUITES += 1))
}

xml_finish_suite() {

    JUNIT_XML="${JUNIT_XML}${NL}  <testsuite name=\"${JUNIT_SUITE_NAME}\" tests=\"${JUNIT_SUITE_TESTS}\" failures=\"${JUNIT_SUITE_FAIL}\" disabled=\"${JUNIT_SUITE_DISABLED}\" error=\"0\" time=\"0\">  ${JUNIT_SUITE_XML}${NL}  </testsuite>"

    JUNIT_SUITE_NAME='UNDEFINED_SUITE'
    JUNIT_SUITE_XML=''  # XML for testcases
    JUNIT_SUITE_TESTS=0
    JUNIT_SUITE_PASS=0
    JUNIT_SUITE_FAIL=0
    JUNIT_SUITE_DISABLED=0
}

disabled_test() {
    JUNIT_SUITE_XML="${JUNIT_SUITE_XML}${NL}    <testcase name=\"${JUNIT_WRAP_TEST}\" status=\"notrun\" time=\"0\" classname=\"${JUNIT_SUITE_NAME}\" />"
    ((JUNIT_DISABLED += 1))
    ((JUNIT_SUITE_DISABLED += 1))
}


# -----------------------------------------------------------------------------
# Make sure necessary environment variables are set up
# -----------------------------------------------------------------------------

if [ -z "${JUNIT_WRAP_FILE}" ] ; then
    ( >&2 echo '***** Error: The required environment variable JUNIT_WRAP_FILE is not set' )
    export JUNIT_WRAP_FILE='junit_wrap_results.xml'
fi

if [ -z "${JUNIT_WRAP_SUITE}" ] ; then
    ( >&2 echo '***** Error: The required environment variable JUNIT_WRAP_SUITE is not set' )
    export JUNIT_WRAP_SUITE='UNDEFINED_TEST'
fi

if [ -z "${JUNIT_WRAP_TEST}" ] ; then
    ( >&2 echo '***** Error: The required environment variable JUNIT_WRAP_TEST is not set' )
    export JUNIT_WRAP_TEST='UNDEFINED_SUITE'
fi


# -----------------------------------------------------------------------------
# Run the command under test
# -----------------------------------------------------------------------------

xml_start_suite "${JUNIT_WRAP_SUITE}"
echo "junit-wrap: Command is [$@]"

eval $@
declare -i STATUS="$?"


# -----------------------------------------------------------------------------
# Write the XML for this command as a single JUnit test
# -----------------------------------------------------------------------------

echo "junit-wrap: Command returned ${STATUS}"

if [ "${STATUS}" = 0 ]
then
    ((JUNIT_TESTS += 1))
    ((JUNIT_PASS += 1))
    ((JUNIT_SUITE_TESTS += 1))
    ((JUNIT_SUITE_PASS += 1))
    JUNIT_SUITE_XML="${JUNIT_SUITE_XML}${NL}    <testcase name=\"${JUNIT_WRAP_TEST}\" status=\"run\" time=\"0\" classname=\"${JUNIT_SUITE_NAME}\" />"
    echo
    echo "SUCCESS"
    echo
    ((NUM_PASSED += 1))
else
    ((JUNIT_TESTS += 1))
    ((JUNIT_FAIL += 1))
    ((JUNIT_SUITE_TESTS += 1))
    ((JUNIT_SUITE_FAIL += 1))
    JUNIT_SUITE_XML="${JUNIT_SUITE_XML}${NL}    <testcase name=\"${JUNIT_WRAP_TEST}\" status=\"run\" time=\"0\" classname=\"${JUNIT_SUITE_NAME}\">${NL}      <failure message=\"Test ${JUNIT_WRAP_TEST} failed\" type=\"\" />${NL}    </testcase>"
    echo
    echo "FAIL (exit code ${STATUS})"
    echo
    ((NUM_FAILED += 1))
fi

xml_finish_suite

# -----------------------------------------------------------------------------
# Write the JUnit results to an XML file
# -----------------------------------------------------------------------------

# Write the XUnit XML file
echo '<?xml version="1.0" encoding="UTF-8"?>' > "${JUNIT_WRAP_FILE}"
echo "<testsuites tests=\"${JUNIT_TESTS}\" failures=\"${JUNIT_FAIL}\" disabled=\"${JUNIT_DISABLED}\" errors=\"0\" time=\"0\">" >> "${JUNIT_WRAP_FILE}"
echo "${JUNIT_XML}" >> "${JUNIT_WRAP_FILE}"
echo '</testsuites>' >> "${JUNIT_WRAP_FILE}"
echo "JUnit XML output written to ${JUNIT_WRAP_FILE}"

# Exit the junit-wrap script with the exact exit code of the unit-test process,
# so that any upstream script properly triggers on the exit code
exit ${STATUS}
