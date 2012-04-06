#--------------------------------------------------------------------------
# File and Version Information:
#  $Id:$
#
# Description:
#  The API for posting messages into E-Log
#
#------------------------------------------------------------------------

"""The API for posting messages into E-Log

This software was developed for the LCLS project.  If you use all or
part of it, please give an appropriate acknowledgment.

@version $Id:$

@author Igor Gaponenko
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision:$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------

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
import tempfile

import urllib
import urllib2

#-----------------------------
# Imports for other modules --
#-----------------------------

# This non-standard package is available from:
#
#   http://atlee.ca/software/poster/index.html
#
# See more information on it from:
#
#   http://stackoverflow.com/questions/680305/using-multipartposthandler-to-post-form-data-with-python

from poster.encode import multipart_encode, MultipartParam
from poster.streaminghttp import register_openers

#----------------------------------
# Local non-exported definitions --
#----------------------------------

default_ws_base_url       = 'https://pswww/pcdsn/ws-auth'  # this may change, so it's better not to rely on this at all
default_ws_login_user     = pwd.getpwuid(os.geteuid())[0]  # assume current logged user
default_ws_login_password = ''

#---------------------
#  Class definition --
#---------------------

class message_poster(object):

    """
    The class to encapsulate user interuction with E-Log via
    Web services.
    """

    # This parameter is fixed. It represent a user posting
    # new entries into E-Log.

    _logbook_author  = pwd.getpwuid(os.geteuid())[0]

    # Parameters provided/calculated in the constructor

    _logbook_instrument = None       # mandatory (constructor)
    _logbook_experiment = 'current'  # mandatory (constructor)
    _logbook_child_cmd  = None       # optional  (constructor)

    # Parameters of the experiment. This data structure will be properly
    # set (if possible) during 'lazy' initialization of the object's context.

    _logbook_experiment_info = None

    # Connection parameters for Web services

    _ws_base_url       = default_ws_base_url
    _ws_login_user     = default_ws_login_user
    _ws_login_password = default_ws_login_password

    # The variable is used by method _init() to perform one time
    # 'lazy' initialization of the internal context of the object.

    _initialized = False

    def __init__(self, instrument, experiment=None, ws_base_url=None, ws_login_user=None, ws_login_password=None, child_cmd=None):

        if instrument:        self._logbook_instrument = instrument
        else:
            raise Exception("no instrument name found among parameters")

        if experiment:        self._logbook_experiment = experiment
        if ws_base_url:       self._ws_base_url        = ws_base_url
        if ws_login_user:     self._ws_login_user      = ws_login_user
        if ws_login_password: self._ws_login_password  = ws_login_password
        if child_cmd:         self._logbook_child_cmd  = child_cmd

    def post(self, message_text, tags=None, attachments=None, parent_message_id=None, run_num=None):

        self._init()

        if parent_message_id and run_num:
            raise Exception("run number can't be used together with the parent message ID")

        child_output = ''
        if self._logbook_child_cmd is not None:
            child_output = os.popen(self._logbook_child_cmd).read()

        if parent_message_id and runnum:
            raise Exception("inconsistent parameters: run number can't be used together with the parent message ID")

        params = [ ('author_account' , self._logbook_author),
                   ('id'             , self._logbook_experiment_info['id']),
                   ('message_text'   , message_text),
                   ('text4child'     , child_output) ]

        if tags:
            params.append( ('num_tags', str(len(tags))) )

            tag_idx = 0 # must begin with 0
            for tag in tags:
                params.extend( [ ("tag_name_%d"  % tag_idx, tag[0]),
                                 ("tag_value_%d" % tag_idx, tag[1]) ] )
                tag_idx = tag_idx + 1
        else:
            params.append(('num_tags', '0'))

        if attachments:
            a_idx = 1 # must begin with 1
            for a in attachments:
                params.extend( [ MultipartParam.from_file("file%d" % a_idx, a[0]),
                                 ("file%d" % a_idx,                         a[1]) ] )
                a_idx = a_idx + 1

        if      run_num:        params.extend( [ ('scope','run'),      ('run_num',   run_num)           ] )
        elif parent_message_id: params.extend( [ ('scope','message'),  ('message_id',parent_message_id) ] )
        else:                   params.extend( [ ('scope','experiment')                                 ] )

        try:

            url = ''.join([self._ws_base_url,'/LogBook/NewFFEntry4grabberJSON.php'])

            datagen,headers = multipart_encode(params)

            req      = urllib2.Request(url, datagen, headers)
            response = urllib2.urlopen(req)
            the_page = response.read()
            result   = simplejson.loads(the_page)
            if result['status'] != 'success':
                raise Exception("failed to interpret server response because of: %s" % str(result['message']))

        except urllib2.URLError, reason:
            raise Exception("failed to post the message due to: %s" % str(reason))

        except urllib2.HTTPError, code:
            raise Exception("failed to post the message due to: %s" % str(code))

 
    def _ws_configure_auth(self):
        """
        Configure authentication context of the web service
        """

        try:

            # First, register openers required by multi-part poster
            #
            opener = register_openers()

            # Then configure and add a handler for Apache Basic Authentication
            #
            password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
            password_mgr.add_password(None, self._ws_base_url, self._ws_login_user, self._ws_login_password)

            opener.add_handler(urllib2.HTTPBasicAuthHandler(password_mgr))

        except urllib2.URLError, reason:
            raise Exception("failed to set up Web Service authentication context due to: ", reason)

        except urllib2.HTTPError, code:
            raise Exception("failed to set up Web Service authentication context due to: ", code)


    def _ws_get_experiment_info (self):
        """
        Contact the Web service an obtain parameters of an experiment specified
        at object creation. If pseudo-experiment name 'current' was used then
        translate the name into the name of the corresponding real experiment
        before proiceedwing with the rest of the operation.
        The real experiment must be known to the Web service, and the user should
        have proper privileges to post messages in a context of the experiment.
        """

        if  self._logbook_experiment == 'current':
            self._logbook_experiment  = self._ws_get_current_experiment()

        # Try both experiments (at instruments) and facilities (at locations)
        #
        urls = [ ''.join([self._ws_base_url,'/LogBook/RequestExperimentsNew.php?instr=',self._logbook_instrument,'&access=post']),
                 ''.join([self._ws_base_url,'/LogBook/RequestExperimentsNew.php?instr=',self._logbook_instrument,'&access=post&is_location' ]) ]

        try:

            for url in urls:

                req      = urllib2.Request(url)
                response = urllib2.urlopen(req)
                the_page = response.read()
                result   = simplejson.loads(the_page)

                for experiment in result['ResultSet']['Result']:
                    if  self._logbook_experiment == experiment['name']:
                        self._logbook_experiment_info = experiment
                        return

            if not self._logbook_experiment_info:
                raise Exception("specified experiment '%s' is either unknown or not available for the user" % self._logbook_experiment)

        except urllib2.URLError, reason:
            raise Exception("failed to get a list of experiment from Web Service due to: ", reason)

        except urllib2.HTTPError, code:
            raise Exception("failed to get a list of experiment from Web Service due to: ", code)



    def _ws_get_current_experiment (self):
        """
        Get the name of the currently active experiment for an instrument
        passed to the contsructor of the class.
        """

        url = ''.join([self._ws_base_url,'/LogBook/RequestCurrentExperiment.php?instr=',self._logbook_instrument])

        try:

            req      = urllib2.Request(url)
            response = urllib2.urlopen(req)
            the_page = response.read()
            result   = simplejson.loads(the_page)
            if len(result) <= 0:
                raise Exception("no experiments are registered for instrument: %s" % self._logbook_instrument)

            experiment = result['ResultSet']['Result']
            if experiment is not None: return experiment['name']

            raise Exception("no current experiment configured for this instrument")

        except urllib2.URLError, reason:
            raise Exception("failed to get the current experiment info from Web Service due to: ", reason)

        except urllib2.HTTPError, code:
            raise Exception("failed to get the current experiment info from Web Service due to: ", code)



    def _init (self):
        """
        Initialize the context, load dictionaries from the web service(s),
        verify parameters and authorizations.
        """

        if self._initialized: return

        # Configure the authentication context for the service

        self._ws_configure_auth()

        # Find the specified experiment and verify caller's privileges to post messages
        # for the experiment.

        self._ws_get_experiment_info()

        self._initialized = True

