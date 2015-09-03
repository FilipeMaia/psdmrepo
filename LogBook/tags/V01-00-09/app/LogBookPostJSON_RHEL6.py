#!/usr/bin/env python

####################################################
# Standard packages from a local Python installation

import httplib
import mimetools
import mimetypes
import os
import os.path
import pwd
import simplejson
import socket
import stat
import sys
import urllib
import urllib2

#############################################
# This non-standard package is available from:
#
#   http://atlee.ca/software/poster/index.html
#
# See more information on it from:
#
#   http://stackoverflow.com/questions/680305/using-multipartposthandler-to-post-form-data-with-python

from poster.encode import multipart_encode, MultipartParam
from poster.streaminghttp import register_openers

######################################
# And here follows the grabber's logic

# ------------------------------------------------------------
# Unspecified values of the LogBook parameters must be set
# either via commad line parameters or inrteractivelly before
# posting the messages.
# ---------------------------------------

logbook_author      = pwd.getpwuid(os.geteuid())[0]  # this is fixed
logbook_instrument  = None                           # command line parameter
logbook_experiment  = None                           # command line parameter (optional)
logbook_use_current = False                          # assume current experiment if set to True
logbook_child_cmd   = None

# The dictionary of available experiments. Experiment names serve as keys.
# This information is equested from the web service. If a specific experiment
# was requested on via a command-line parameter then there will be only
# one entry in the dictionary.
#
# NOTE: Once initialized the dictionary must have at least one entry.

logbook_experiments = None

# ------------------------------------------------------------------
# Default values of these parameters can be changed via command line
# option of the application.
# ------------------------------------------------------------------

ws_url            = None
ws_login_user     = pwd.getpwuid(os.geteuid())[0]
ws_login_password = 'pcds'

message2post = ''
file2attach = None
tag = 'COMMAND_LINE_POST'

# ------------------------------------
# The function for posting a new entry
# ------------------------------------

def post_entry():

    exper_id = logbook_experiments[logbook_experiment]['id']

    url = ws_url+'/LogBook/NewFFEntry4grabberJSON.php'

    child_output = ''
    if logbook_child_cmd is not None: child_output = os.popen(logbook_child_cmd).read()

    if file2attach is None:
        datagen,headers = multipart_encode([
            ('author_account', logbook_author),
            ('id', exper_id),
            ('message_text', message2post),
            ('scope', 'experiment'),
            ('num_tags', '1'),
            ('tag_name_0',tag),
            ('tag_value_0',''),
            ('text4child', child_output) ])

    else:
        idx = file2attach.rfind('/')
        if idx == -1: file2attach_descr = file2attach
        else:         file2attach_descr = file2attach[idx+1:]
        datagen,headers = multipart_encode([
            ('author_account', logbook_author),
            ('id', exper_id),
            ('message_text', message2post),
            ('scope', 'experiment'),
            ('num_tags', '1'),
            ('tag_name_0',tag),
            ('tag_value_0',''),
            ('text4child', child_output),
            MultipartParam.from_file("file1", file2attach),
            ("file1", file2attach_descr) ])

    try:
        req = urllib2.Request(url, datagen, headers)
        response = urllib2.urlopen(req)
        the_page = response.read()
        result = simplejson.loads(the_page)
        if result['status'] != 'success':
            print "Error", result['message']
            sys.exit(1)
        print 'New message ID:', int(result['message_id'])

    except urllib2.URLError, reason:
        print  "Submit New Message Error", reason
        sys.exit(1)

    except urllib2.HTTPError, code:
        print  "Submit New Message Error", code
        sys.exit(1)

 
# ------------------------------------------------------
# Configure an authentication context of the web service
# ------------------------------------------------------

def ws_configure_auth():

    try:

        # First, register openers required by multi-part poster
        #
        opener = register_openers()

        # Then configure and add a handler for Apache Basic Authentication
        #
        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, ws_url, ws_login_user, ws_login_password)

        opener.add_handler(urllib2.HTTPBasicAuthHandler(password_mgr))

    except urllib2.URLError, reason:
        print "ERROR: failed to set up Web Service authentication context due to: ", reason
        sys.exit(1)
    except urllib2.HTTPError, code:
        print "ERROR: failed to set up Web Service authentication context due to: ", code
        sys.exit(1)

# ------------------------------------------------------

def ws_get_experiments (experiment=None):

    # Try both experiments (at instruments) and facilities (at locations)
    #
    urls = [ ws_url+'/LogBook/RequestExperimentsNew.php?instr='+logbook_instrument+'&access=post',
             ws_url+'/LogBook/RequestExperimentsNew.php?instr='+logbook_instrument+'&access=post&is_location' ]

    try:

        d = dict()

        for url in urls:

            req = urllib2.Request(url)
            response = urllib2.urlopen(req)
            the_page = response.read()
            result = simplejson.loads(the_page)
            if len(result) <= 0:
                print "ERROR: no experiments are registered for instrument: %s" % logbook_instrument

            # if the experiment was explicitly requested in the command line then try to find
            # the one. Otherwise return the whole list
            #
            if experiment is not None:
                for e in result['ResultSet']['Result']:
                    if experiment == e['name']:
                        d[experiment] = e
            else:
                for e in result['ResultSet']['Result']:
                    d[e['name']] = e

        return d

    except urllib2.URLError, reason:
        print "ERROR: failed to get a list of experiment from Web Service due to: ", reason
        sys.exit(1)
    except urllib2.HTTPError, code:
        print "ERROR: failed to get a list of experiment from Web Service due to: ", code
        sys.exit(1)

def ws_get_current_experiment ():

    url = ws_url+'/LogBook/RequestCurrentExperiment.php?instr='+logbook_instrument

    try:

        req      = urllib2.Request(url)
        response = urllib2.urlopen(req)
        the_page = response.read()
        result   = simplejson.loads(the_page)
        if len(result) <= 0:
            print "ERROR: no experiments are registered for instrument: %s" % logbook_instrument

        e = result['ResultSet']['Result']
        if e is not None:
            return e['name']

        print "ERROR: no current experiment configured for this instrument"
        sys.exit(1)

    except urllib2.URLError, reason:
        print "ERROR: failed to get the current experiment info from Web Service due to: ", reason
        sys.exit(1)
    except urllib2.HTTPError, code:
        print "ERROR: failed to get the current experiment info from Web Service due to: ", code
        sys.exit(1)

# -----------------------
# Application starts here
# -----------------------

if __name__ == "__main__" :

    # -----------------------------
    # Parse command line parameters
    # -----------------------------

    import getopt

    def help(progname):
        print """
DESCRIPTION:

  This application allows composing and posting messages containing screenshots
  of X11 windows or regions into Electornic LogBook of LCLS experiment.

USAGE:

  %s -i <instrument> -e <experiment> -w <web-service-url> -u <web-service-user> -p <web-service-password> -m <message2post> -a <file2attach> -t <tag>
  %s -i <instrument> -e current      -w <web-service-url> -u <web-service-user> -p <web-service-password> -m <message2post> -a <file2attach> -t <tag>

OPTIONS & PARAMETERS:

  ___________________
  Required parameters

    -i <instrument>

      the name of an instrument

    -w web-service-url

      the base URL of the LogBook Web Service. 

  ___________________
  Optional parameters

    -e <experiment>
    -e current

      the name of some specific experiment or 'current' as an indication
      to the currently active experiment. If neither name is provided
      the grabber will contact the LogBook Web service to request
      the names of all experiments associated with the above
      selected instrument. In the later case it will be up to
      a user of the Grabber to select a desired experiment.

    -u web-service-user

      the user name to connect to the service (NOTE: this
      is not the same account the current application is
      being run from!)

    -p web-service-password

      the password to connect to the service. Its value
      is associated with the above mentioned service account.

    -m message2post

      an optional text string to be posted in the new e-log entry

    -a file2attach

      an optional file to be attached by the entry

    -t tag

      an optional tag for the entry

    -c command-for-child-message

      allows posting a child message with a standard output of the specified
      UNIX command (or an application) found after the option. The child
      will be posted immediattely after the parrent message.

      NOTE: using this option may delay the time taken by the Grabber
            by a duration of time needed by the specified command to produce
            its output

""" % (progname)

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:e:w:u:p:m:a:t:c:')
    except getopt.GetoptError, errmsg:
        print "ERROR:", errmsg
        sys.exit(1)

    for name, value in opts:
        if name in ('-h',):
            help(sys.argv[0])
            sys.exit(0)
        elif name in ('-i',):
            logbook_instrument = value
        elif name in ('-e',):
            logbook_experiment = value
        elif name in ('-w',):
            ws_url = value
        elif name in ('-u',):
            ws_login_user = value
        elif name in ('-p',):
            ws_login_password = value
        elif name in ('-m',):
            message2post = value
        elif name in ('-a',):
            file2attach = value
        elif name in ('-t',):
            tag = value
        elif name in ('-c',):
            logbook_child_cmd = value
        else:
            print "invalid argument:", name
            sys.exit(2)

    if logbook_instrument is None:
        print "no instrument name found among command line parameters"
        sys.exit(3)

    if logbook_experiment is None:
        print "no experiment name found among command line parameters"
        sys.exit(3)

    if ws_url is None:
        print "no web service URL found among command line parameters"
        sys.exit(3)

    # ----------------------------------------------------
    # Configure the authentication context for the service
    # ----------------------------------------------------

    ws_configure_auth ()

    # ---------------------------------------------------------
    # If the current experiment was requested then check what's
    # (if any) the current experiment for the instrument.
    # Otherwise get a list of experiments for the specified instrument
    # and make sure the requested experiment is among those.
    # ---------------------------------------------------------

    logbook_experiments = ws_get_experiments()

    if logbook_experiment == 'current':
        logbook_experiment = ws_get_current_experiment ()
    else:
        if logbook_experiments[logbook_experiment] is None:
            print "no such experiment found in this instrument"
            sys.exit(3)

    post_entry()

    sys.exit(0)
