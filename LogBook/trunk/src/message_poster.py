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

default_ws_base_url       = 'https://pswww.slac.stanford.edu/ws-auth/'  # this may change, so it's better not to rely on this at all
default_ws_login_user     = pwd.getpwuid(os.geteuid())[0]               # assume current logged user
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

    _logbook_instrument = None  # mandatory (constructor)
    _logbook_experiment = None  # mandatory (constructor)
    _logbook_child_cmd  = None  # optional  (constructor)

    # Connection parameters for Web services

    _ws_base_url       = default_ws_base_url
    _ws_login_user     = default_ws_login_user
    _ws_login_password = default_ws_login_password

    # The variable is used by method _init() to perform one time
    # 'lazy' initialization of the internal context of the object.

    _initialized = False

    def __init__(self, instrument, experiment, ws_base_url=None, ws_login_user=None, ws_login_password=None, child_cmd=None):

        if instrument:        self._logbook_instrument = instrument
        else:
            raise Exception("no instrument name found among parameters")

        if experiment:        self._logbook_experiment = experiment
        if ws_base_url:       self._ws_base_url        = ws_base_url
        if ws_login_user:     self._ws_login_user      = ws_login_user
        if ws_login_password: self._ws_login_password  = ws_login_password
        if child_cmd:         self._logbook_child_cmd  = child_cmd

    def post(self, message_text, tags=None, attachments=None, parent_message_id=None, run_num=None):

        """
        ____________
        ATTACHMENTS:

          Attachments are specified as a tuple/list of entries:

            attachments: [entry1, entry2, ...]

          Where each entry can be eather a path to a file to be attached:

            entry: '/u2/tmp/MyFile.jpg'

          or a tuple/list of 1 or 2 elements:

            entry: ('/u2/tmp/MyFile.jpg','Name2displayInELog')
            entry: ('/u2/tmp/MyFile.jpg','')

          The second optional element represents a name which will be displayed
          in e-Log for the corresponding attachment. If no name is provided or
          if it's an empty string then the file name w/o its directory will be
          assumed.

          When specifying multiple attachments strings and tuples can be freely
          mixed. For example, consider adding 5 atatchments:

            [ '/u2/tmp/MyFile.jpg',
              ('/u2/tmp/MyFile.jpg','Name2displayInELog'),
              ['/u2/tmp/MyFile2.jpg',''],
              'SomethingElse.pdf',
              ('./Image.jpg',)
            ]

        _____
        TAGS:

          Tags are specified similarly to atatchments:


            tags: [entry1, entry2, ...]

          Where each entry can be eather the name of a tag:

            entry: 'MyTag'

          or a tuple/list of 1 or 2 elements:

            entry: ('MyTag_with_value','some_value')
            entry: ('MyTag_without_value',)

          Note that tags in e-log are allowed to have optional values. Usage of
          values depends on a specific Web interface to e-log. Most tags in the
          current e-log don't have values.

        """
        self._init()

        if parent_message_id and run_num:
            raise Exception("run number can't be used together with the parent message ID")

        child_output = ''
        if self._logbook_child_cmd is not None:
            child_output = os.popen(self._logbook_child_cmd).read()

        if parent_message_id and run_num:
            raise Exception("inconsistent parameters: run number can't be used together with the parent message ID")

        params = [ ('author_account' , self._logbook_author),
                   ('instrument'     , self._logbook_instrument),
                   ('experiment'     , self._logbook_experiment),
                   ('message_text'   , message_text),
                   ('text4child'     , child_output) ]

        if tags:
            params.append( ('num_tags', str(len(tags))) )

            tag_idx = 0 # must begin with 0
            for tag in tags:
                t_name = None
                t_value = ''
                if isinstance(tag,str):
                    t_name = tag
                elif isinstance(tag,tuple) or isinstance(tag,list):
                    t_name = tag[0]
                    if len(tag) > 1: t_value = tag[1]
                else:
                    raise ValueError("illegal type of the tag parameter")

                params.extend( [ ("tag_name_%d"  % tag_idx, t_name),
                                 ("tag_value_%d" % tag_idx, t_value) ] )
                tag_idx = tag_idx + 1
        else:
            params.append(('num_tags', '0'))

        if attachments:
            a_idx = 1 # must begin with 1
            for a in attachments:
                a_path = None
                a_name = ''
                if isinstance(a,str):
                    a_path = a
                elif isinstance(a,tuple) or isinstance(a,list):
                    a_path = a[0]
                    if len(a) > 1: a_name = a[1]
                else:
                    raise ValueError("illegal type of the attachment parameter")

                if a_name == '':
                    idx = a_path.rfind('/')
                    if idx == -1: a_name = a_path
                    else:         a_name = a_path[idx+1:]

                params.extend( [ MultipartParam.from_file("file%d" % a_idx, a_path),
                                 ("file%d" % a_idx,                         a_name) ] )
                a_idx = a_idx + 1

        if      run_num:        params.extend( [ ('scope','run'),      ('run_num',   run_num)           ] )
        elif parent_message_id: params.extend( [ ('scope','message'),  ('message_id',parent_message_id) ] )
        else:                   params.extend( [ ('scope','experiment')                                 ] )

        try:

            url = ''.join([self._ws_base_url,'/LogBook/NewFFEntry4posterJSON.php'])

            datagen,headers = multipart_encode(params)

            req      = urllib2.Request(url, datagen, headers)
            response = urllib2.urlopen(req)
            the_page = response.read()
            result   = simplejson.loads(the_page)
            if result['status'] != 'success':
                raise Exception("failed to interpret server response because of: %s" % str(result['message']))

            return int(result['message_id'])

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

        # Find the actual name of the 'current' experiment in case if that
        # pseudo-experiment name was passed into the object's constructor.

        if  self._logbook_experiment == 'current':
            self._logbook_experiment  = self._ws_get_current_experiment()


        self._initialized = True


class message_poster_self(message_poster):

    """
    The class to encapsulate user interuction with E-Log via
    Web services. Unlike its base class message_poster this one
    would post messages on behalf of the current logged user w/o
    a need to provide Web server credentials explicitly.
    """

    def __init__(self, instrument, experiment, ws_base_url=None, child_cmd=None):

        message_poster.__init__(self, instrument, experiment, ws_base_url, ws_login_user='amoopr', ws_login_password='pcds',child_cmd=child_cmd)