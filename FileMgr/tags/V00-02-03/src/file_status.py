#--------------------------------------------------------------------------
# File and Version Information:
#  $Id:$
#
# Description:
#  The API for inquering file status from the Data Manager
#
#------------------------------------------------------------------------

"""The API for inquering file status from the Data Manager

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

default_ws_base_url       = 'https://pswww.slac.stanford.edu/pcdsn/ws-auth'  # this may change, so it's better not to rely on this at all
default_ws_login_user     = pwd.getpwuid(os.geteuid())[0]  # assume current logged user
default_ws_login_password = ''

#------------------------
# Exported definitions --
#------------------------

# File status flags

IN_MIGRATION_DATABASE = 'IN_MIGRATION_DATABASE'
DISK = 'DISK'
HPSS = 'HPSS'

class ExtendedStatus(object):
    def __init__(self,flags,size_bytes):
        self._flags = flags
        self._size_bytes = size_bytes
    def flags(self): return self._flags
    def size_bytes(self): return self._size_bytes

#---------------------
#  Class definition --
#---------------------

class file_status(object):

    """
    The class to encapsulate user interuction with Data Manager via
    Web services.
    """

    # Connection parameters for Web services

    _ws_base_url       = default_ws_base_url
    _ws_login_user     = default_ws_login_user
    _ws_login_password = default_ws_login_password

    # The variable is used by method _init() to perform one time
    # 'lazy' initialization of the internal context of the object.

    _initialized = False

    def __init__(self, ws_base_url=None, ws_login_user=None, ws_login_password=None):

        if ws_base_url:       self._ws_base_url       = ws_base_url
        if ws_login_user:     self._ws_login_user     = ws_login_user
        if ws_login_password: self._ws_login_password = ws_login_password

    def filter(self, files2check, evaluator):
        """
        The function will obtain an extended status of each file specified in
        the input list 'files2check' and let a user defined function 'evaluator'
        to decide if the file has to be included into the resulting list to be
        returned to a caller.

        PARAMETERS:

          files2check: is a list of triplets (experiment_id,file_type,file_name)
                       Note that experiment_id has to be a number, file_type
                       has to be one of the types known to the Data Manager,
                       and file_name should not contain any path.

          evaluator:   is a user defined function which is supposed to have the
                       following signature:

                         evaluator(triplet,status)

                       The function shall return True if the file should be included
                       into resulting list, or False otherwise.

                       The 'triplet' parameter will be a file entry from the input
                       list. The 'status' object will represent the extended status
                       of the file, which will include as minimum the following
                       methods:

                         flags(): a list of file status flags earlier in the header section
                                  of this file.

                         size_bytes(): the file size in bytes. Note that 0 value may mean
                                       no information available for a file (file may not be
                                       registered in the File Manager Catalog, etc.)

        RETURN:

          a list of triples for files which were successfully selected
          by the user-defined evaluator. The returned list will be a subset
          of the input one.

        EXCEPTIONS:

          the function may generate exceptions of class Exception

        """

        self._init()

        # Evaluate a type and a structuere of the input parameters.
        #
        if type(files2check) not in (list,tuple):
            raise Exception("parameter files2check  has to be a list of triplets [(experiment_id,file_type,file_name),..]")
        for triplet in files2check:
            if type(triplet) not in (list,tuple):
                raise Exception("files2check has at least one element which is not a triple of (experiment_id,file_type,file_name)")
            if len(triplet) != 3:
                raise Exception("files2check has at least one element which doesn'r have 3 components (experiment_id,file_type,file_name)")

        try:

            url = ''.join([self._ws_base_url,'/FileMgr/GetExtendedFileStatusJSON.php'])

            params = [('files',simplejson.dumps(files2check))]

            datagen,headers = multipart_encode(params)

            req      = urllib2.Request(url, datagen, headers)
            response = urllib2.urlopen(req)
            the_page = response.read()
            result   = simplejson.loads(the_page)
            if result['status'] != 'success':
                raise Exception("Web service reported error: %s" % str(result['message']))

            # Now let the user-defined evaluator to check
            #
            selected_files = []

            for entry in result['files_extended']:

                triplet = entry[0]
                status  = ExtendedStatus(entry[1],int(entry[2]))

                if evaluator(triplet,status):
                    selected_files.append(triplet)

            return selected_files

        except urllib2.URLError, reason:
            raise Exception("failed to submit the request to a Web Service due to: %s" % str(reason))

        except urllib2.HTTPError, code:
            raise Exception("failed to submit the request to a Web Service due to: %s" % str(code))

 
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

    def _init (self):
        """
        Initialize the context, load dictionaries from the web service(s) if needed,
        verify parameters and authorizations.
        """

        if self._initialized: return

        # Configure the authentication context for the service

        self._ws_configure_auth()

        self._initialized = True

