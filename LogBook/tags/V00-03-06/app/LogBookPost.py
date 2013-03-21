#!/usr/bin/env python

###################################################
# This is the patch to th estandard package urllib2
###################################################

################################################################################
# Version: 0.2.1
#  - StringIO workaround (Laurent Coustet), does not work with cStringIO
#
# Version: 0.2.0
#  - UTF-8 filenames are now allowed (Eli Golovinsky)
#  - File object is no more mandatory, Object only needs to have seek() read() attributes (Eli Golovinsky)
#
# Version: 0.1.0
#  - upload is now done with chunks (Adam Ambrose)
#
# Version: older
# THANKS TO:
# bug fix: kosh @T aesaeion.com
# HTTPS support : Ryan Grow <ryangrow @T yahoo.com>

# Copyright (C) 2004,2005,2006,2008,2009 Fabien SEISEN
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# you can contact me at: <fabien@seisen.org>
# http://fabien.seisen.org/python/
#
# Also modified by Adam Ambrose (aambrose @T pacbell.net) to write data in
# chunks (hardcoded to CHUNK_SIZE for now), so the entire contents of the file
# don't need to be kept in memory.
#
"""
enable to upload files using multipart/form-data

idea from:
upload files in python:
 http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/146306

timeoutsocket.py: overriding Python socket API:
 http://www.timo-tasi.org/python/timeoutsocket.py
 http://mail.python.org/pipermail/python-announce-list/2001-December/001095.html

import urllib2_files
import urllib2
u = urllib2.urlopen('http://site.com/path' [, data])

data can be a mapping object or a sequence of two-elements tuples
(like in original urllib2.urlopen())
varname still need to be a string and
value can be string of a file object
eg:
  ((varname, value),
   (varname2, value),
  )
  or
  { name:  value,
    name2: value2
  }

"""

import httplib
import mimetools
import mimetypes
import os
import os.path
import socket
import stat
import sys
import urllib
import urllib2

CHUNK_SIZE = 65536

def get_content_type(filename):
    return mimetypes.guess_type(filename)[0] or 'application/octet-stream'

# if sock is None, return the estimate size
def send_data(v_vars, v_files, boundary, sock=None):
    l = 0
    for (k, v) in v_vars:
        buffer=''
        buffer += '--%s\r\n' % boundary
        buffer += 'Content-Disposition: form-data; name="%s"\r\n' % k
        buffer += '\r\n'
        buffer += v + '\r\n'
        if sock:
            sock.send(buffer)
        l += len(buffer)
    for (k, v) in v_files:
        fd = v
    	# Special case for StringIO
        #if fd.__module__ in ("StringIO", "cStringIO"):
        #    name = k
        #    fd.seek(0, 2) # EOF
        #    file_size = fd.tell()
        #    fd.seek(0) # START
        #else:
        file_size = os.fstat(fd.fileno())[stat.ST_SIZE]
        name = fd.name.split('/')[-1]
        if isinstance(name, unicode):
            name = name.encode('UTF-8')
        buffer=''
        buffer += '--%s\r\n' % boundary
        buffer += 'Content-Disposition: form-data; name="%s"; filename="%s"\r\n' \
                  % (k, name)
        buffer += 'Content-Type: %s\r\n' % get_content_type(name)
        buffer += 'Content-Length: %s\r\n' % file_size
        buffer += '\r\n'

        l += len(buffer)
        if sock:
            sock.send(buffer)
            if hasattr(fd, 'seek'):
                fd.seek(0)
        while True:
            chunk = fd.read(CHUNK_SIZE)
            if not chunk:
                break
            if sock:
                sock.send(chunk)
        l += file_size
    buffer='\r\n'
    buffer += '--%s--\r\n' % boundary
    buffer += '\r\n'
    if sock:
        sock.send(buffer)
    l += len(buffer)
    return l

# mainly a copy of HTTPHandler from urllib2
class newHTTPHandler(urllib2.BaseHandler):
    def http_open(self, req):
        return self.do_open(httplib.HTTP, req)

    def do_open(self, http_class, req):
        data = req.get_data()
        v_files=[]
        v_vars=[]
        # mapping object (dict)
        if req.has_data() and type(data) != str:
            if hasattr(data, 'items'):
                data = data.items()
            else:
                try:
                    if len(data) and not isinstance(data[0], tuple):
                        raise TypeError
                except TypeError:
                    ty, va, tb = sys.exc_info()
                    raise TypeError, "not a valid non-string sequence or mapping object", tb

            for (k, v) in data:
                if hasattr(v, 'read'):
                    v_files.append((k, v))
                else:
                    v_vars.append( (k, v) )
        # no file ? convert to string
        if len(v_vars) > 0 and len(v_files) == 0:
            data = urllib.urlencode(v_vars)
            v_files=[]
            v_vars=[]
        host = req.get_host()
        if not host:
            raise urllib2.URLError('no host given')
        h = http_class(host) # will parse host:port
        if req.has_data():
            h.putrequest('POST', req.get_selector())
            if not 'Content-type' in req.headers:
                if len(v_files) > 0:
                    boundary = mimetools.choose_boundary()
                    l = send_data(v_vars, v_files, boundary)
                    h.putheader('Content-Type',
                                'multipart/form-data; boundary=%s' % boundary)
                    h.putheader('Content-length', str(l))
                else:
                    h.putheader('Content-type',
                                'application/x-www-form-urlencoded')
                    if not 'Content-length' in req.headers:
                        h.putheader('Content-length', '%d' % len(data))
        else:
            h.putrequest('GET', req.get_selector())

        scheme, sel = urllib.splittype(req.get_selector())
        sel_host, sel_path = urllib.splithost(sel)
        h.putheader('Host', sel_host or host)
        for name, value in self.parent.addheaders:
            name = name.capitalize()
            if name not in req.headers:
                h.putheader(name, value)
        for k, v in req.headers.items():
            h.putheader(k, v)
        # httplib will attempt to connect() here.  be prepared
        # to convert a socket error to a URLError.
        try:
            h.endheaders()
        except socket.error, err:
            raise urllib2.URLError(err)

        if req.has_data():
            if len(v_files) >0:
                l = send_data(v_vars, v_files, boundary, h)
            elif len(v_vars) > 0:
                # if data is passed as dict ...
                data = urllib.urlencode(v_vars)
                h.send(data)
            else:
                # "normal" urllib2.urlopen()
                h.send(data)

        code, msg, hdrs = h.getreply()
        fp = h.getfile()
        if code == 200:
            resp = urllib.addinfourl(fp, hdrs, req.get_full_url())
            resp.code = code
            resp.msg = msg
            return resp
        else:
            return self.parent.error('http', req, fp, code, msg, hdrs)

urllib2._old_HTTPHandler = urllib2.HTTPHandler
urllib2.HTTPHandler = newHTTPHandler

class newHTTPSHandler(newHTTPHandler):
    def https_open(self, req):
        return self.do_open(httplib.HTTPS, req)

urllib2.HTTPSHandler = newHTTPSHandler



######################################
# And here follows th egrabber's logic

import os, pwd
import simplejson

# ------------------------------------------------------------
# Unspecified values of the LogBook parameters must be set
# either via commad line parameters or inrteractivelly before
# posting the messages.
# ---------------------------------------

logbook_author      = pwd.getpwuid(os.geteuid())[0]  # this is fixed
logbook_instrument  = None                           # command line parameter
logbook_experiment  = None                           # command line parameter (optional)
logbook_use_current = False                          # assume current experiment if set to True

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

    url = ws_url+'/LogBook/NewFFEntry4grabber.php'

    if file2attach is None:
        data = (
            ('author_account', logbook_author),
            ('id', exper_id),
            ('message_text', message2post),
            ('scope', 'experiment'),
            ('num_tags', '1'),
            ('tag_name_0',tag),
            ('tag_value_0',''))

    else:
        idx = file2attach.rfind('/')
        if idx == -1: file2attach_descr = file2attach
        else:         file2attach_descr = file2attach[idx+1:]
        data = (
            ('author_account', logbook_author),
            ('id', exper_id),
            ('message_text', message2post),
            ('scope', 'experiment'),
            ('num_tags', '1'),
            ('tag_name_0',tag),
            ('tag_value_0',''),
            ("file1", open( file2attach, 'r' )),
            ("file1", file2attach_descr) )

    try:
        req = urllib2.Request(url, data, {})
        response = urllib2.urlopen(req)
        the_page = response.read()
        if the_page != '':
            print "Error", the_page
            sys.exit(1)

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

        # ---------------------------------------
        # create and configure a password manager
        # ---------------------------------------

        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()

        # Add the username and password.
        # If we knew the realm, we could use it instead of `None`.
        #
        password_mgr.add_password(None, ws_url, ws_login_user, ws_login_password)

        handler = urllib2.HTTPBasicAuthHandler(password_mgr)

        # create "opener" (OpenerDirector instance)
        opener = urllib2.build_opener(handler)

        # Install the opener.
        # Now all calls to urllib2.urlopen use our opener.
        #
        urllib2.install_opener(opener)

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

""" % (progname)

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:e:w:u:p:m:a:t:')
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
