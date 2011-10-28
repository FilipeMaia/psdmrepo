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

import os, pwd, tempfile
import simplejson
import tkMessageBox

from Tkinter import *
from ScrolledText import *

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
# was requestd on via a command-line parametyer then there will be only
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

# -------------------------------------------------
# The class to encapsulate user interuction via GUI
# -------------------------------------------------

class LogBookGrabberUI:

    def __init__(self, parent):

        self.grab = Button(parent, text="Grab", command=self.on_grab)
        self.grab.grid(row=0, column=1,padx=5,pady=5,sticky=W)

        self.submit = Button(parent, text="Submit", command=self.on_submit)
        self.submit.grid(row=0, column=2,padx=5,pady=5,sticky=W)

        self.cancel = Button(parent, text="Cancel",  command=self.on_cancel)
        self.cancel.grid(row=0, column=3,padx=5,pady=5,sticky=W)

        Label(parent, text="Message: ").grid(row=1,column=0,padx=5,pady=5,sticky=W+N)
        self.message = ScrolledText(parent, width=48,height=6)
        self.message.grid(row=1,column=1,columnspan=4,rowspan=4,padx=5,pady=5,sticky=W+N)

        f1 = Frame(parent)
        Label(f1,text="Experiment:").grid(row=0,sticky=W+N)
        f2 = Frame(f1)
        if logbook_experiment is None:
            scrollbar = Scrollbar(f2, orient=VERTICAL)
            self.experiment = Listbox(f2, yscrollcommand=scrollbar.set,height=2)
            scrollbar.config(command=self.experiment.yview)
            scrollbar.grid(row=1,column=4,padx=5,pady=0,sticky=W+N)
            for e in logbook_experiments.keys():
                self.experiment.insert(END, e)
            self.experiment.select_set(0)
            self.experiment.grid(row=1,column=3,padx=5,pady=0,sticky=W+N)
        else:
            self.exp_name = StringVar()
            self.exp_name.set(logbook_experiment)
            Label(f2,textvariable=self.exp_name).pack()
        f2.grid(row=0,column=1,padx=2,sticky=W)
        if logbook_use_current or logbook_experiment is None:
            self.refresh = Button(f1,text="Update", command=self.on_refresh)
            self.refresh.grid(row=2,column=1,padx=5,pady=5,sticky=W)
        f1.grid(row=1,column=5,columnspan=2,rowspan=4,padx=5,pady=5,sticky=W+N)

        Label(parent, text="Author: ").grid(row=6,column=0,padx=5,pady=0,sticky=W+N)
        self.author = Label(parent, text=logbook_author)
        self.author.grid(row=6, column=1,padx=5,pady=0,sticky=W+N)

        Label(parent, text="Run number: ").grid(row=7,column=0,padx=5,pady=5,sticky=W+N)
        self.run = Entry(parent)
        self.run.insert(0, "")
        self.run.grid(row=7, column=1,columnspan=1,padx=5,pady=5,sticky=W+N)
        Label(parent, text="<- use if posting for a specific run").grid(row=7,column=2,columnspan=4,padx=5,pady=5,sticky=W+N)

        Label(parent, text="Message ID: ").grid(row=8,column=0,padx=5,pady=5,sticky=W+N)
        self.message_id = Entry(parent)
        self.message_id.insert(0, "")
        self.message_id.grid(row=8, column=1,columnspan=1,padx=5,pady=5,sticky=W+N)
        Label(parent, text="<- use if replying to an existing message").grid(row=8,column=2,columnspan=4,padx=5,pady=5,sticky=W+N)

        Label(parent, text="Description: ").grid(row=9,column=0,padx=5,pady=5,sticky=W+N)
        self.descr = Entry(parent)
        self.descr.insert(0, "Image #1")
        self.descr.grid(row=9, column=1,columnspan=2,padx=5,pady=5,sticky=W+N)


        self.canvas = Canvas(background='black',width=0,height=0)
        self.canvas.grid(row=10,column=0,padx=5,pady=0,columnspan=6,sticky=W+N+E+S)

        self.submit.configure(state=DISABLED)
        self.cancel.configure(state=DISABLED)

    def on_grab(self):

        self.f_jpeg = tempfile.NamedTemporaryFile(mode='r+b',suffix='.jpeg')
        f_gif = tempfile.NamedTemporaryFile(mode='r+b',suffix='.gif')
        print self.f_jpeg.name, f_gif.name
        if( 0 == os.system("import -trim -border %s; convert %s %s" % (self.f_jpeg.name, self.f_jpeg.name, f_gif.name))):
            photo = PhotoImage(file=f_gif.name)

            p_width = photo.width()
            p_height = photo.height()

            max_width  = 800
            max_height = 600

            scale = 1
            if( p_width > max_width ):
                x_scale = 1 + p_width / max_width
                if x_scale > scale: scale = x_scale

            if( p_height > max_height ):
                y_scale = 1 + p_height / max_height
                if y_scale > scale: scale = y_scale

            print scale
            photo1 = photo.subsample(scale)

            self.canvas.configure(width=photo1.width()+10)
            self.canvas.configure(height=photo1.height()+10)

            image = self.canvas.create_image(5,5,anchor=NW,image=photo1,state=NORMAL)

            self.grab.configure(state=DISABLED)
            self.submit.configure(state=NORMAL)
            self.cancel.configure(state=NORMAL)
            image.draw()

    def on_cancel(self):

        self.message.delete("0.0",END)
        self.run.delete(0,last=END)
        self.message_id.delete(0,last=END)

        self.grab.configure(state=NORMAL)
        self.submit.configure(state=DISABLED)
        self.cancel.configure(state=DISABLED)
        self.canvas.configure(width=0)
        self.canvas.configure(height=0)
        self.canvas.delete(ALL)

    def on_submit(self):

        # If a specific experiment was provided then assume the one. Otherwise
        # let a user to select the one from the listbox.
        #
        exper_name = logbook_experiment
        if logbook_experiment is None:

            # Get the experiment identifier from the current selection
            #
            sel = self.experiment.curselection()
            if len(sel) != 1:
                tkMessageBox.showerror (
                    "More Information requested",
                    "No experiment selected" )
                return
            exper_name = self.experiment.get( sel[0])

        exper_id = logbook_experiments[exper_name]['id']

        url = ws_url+'/LogBook/NewFFEntry4grabber.php'
        run = self.run.get()
        message_id = self.message_id.get()

        if(( run != '' ) and ( message_id != '' )):
            tkMessageBox.showerror (
                "Inconsistent Input",
                "Run number can't be used togher with the parent message ID. Choose the right context to post the screenshot and try again." )
            return
        if( run != '' ):
            data = (
                ('author_account', logbook_author),
                ('id', exper_id),
                ('message_text', self.message.get("0.0",END)),
                ('scope', 'run'),
                ('run_num',run),
                ('num_tags', '1'),
                ('tag_name_0','SCREENSHOT'),
                ('tag_value_0',''),
                ("file1", open( self.f_jpeg.name, 'r' )),
                ("file1", self.descr.get()) )
        elif( message_id != '' ): 
            data = (
                ('author_account', logbook_author),
                ('id', exper_id),
                ('message_text', self.message.get("0.0",END)),
                ('scope', 'message'),
                ('message_id', message_id),
                ('num_tags', '1'),
                ('tag_name_0','SCREENSHOT'),
                ('tag_value_0',''),
                ("file1", open( self.f_jpeg.name, 'r' )),
                ("file1", self.descr.get()) )
        else:
            data = (
                ('author_account', logbook_author),
                ('id', exper_id),
                ('message_text', self.message.get("0.0",END)),
                ('scope', 'experiment'),
                ('num_tags', '1'),
                ('tag_name_0','SCREENSHOT'),
                ('tag_value_0',''),
                ("file1", open( self.f_jpeg.name, 'r' )),
                ("file1", self.descr.get()) )

        try:
            req = urllib2.Request(url, data, {})
            response = urllib2.urlopen(req)
            the_page = response.read()
            if the_page != '':
                tkMessageBox.showerror ( "Error", the_page )

        except urllib2.URLError, reason:
            tkMessageBox.showerror (
                "Submit New Message Error",
                reason )
        except urllib2.HTTPError, code:
            tkMessageBox.showerror (
                "Submit New Message Error",
                code )

        self.message.delete("0.0",END)
        self.run.delete(0,last=END)
        self.message_id.delete(0,last=END)
        self.grab.configure(state=NORMAL)
        self.submit.configure(state=DISABLED)
        self.cancel.configure(state=DISABLED)
        self.canvas.configure(width=0)
        self.canvas.configure(height=0)
        self.canvas.delete(ALL)

    def on_refresh (self):

        global logbook_experiment
        global logbook_experiments

        if logbook_use_current:
            logbook_experiment = ws_get_current_experiment ()
            logbook_experiments = ws_get_experiments (logbook_experiment)
            self.exp_name.set(logbook_experiment)
        else:
            logbook_experiments = ws_get_experiments ()
            self.experiment.delete(0,END) 
            for e in logbook_experiments.keys():
                self.experiment.insert(END, e)
            self.experiment.select_set(0)
 
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

  %s -i <instrument> -e <experiment> -w <web-service-url> -u <web-service-user> -p <web-service-password>
  %s -i <instrument> -e current      -w <web-service-url> -u <web-service-user> -p <web-service-password>

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

""" % (progname)

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:e:w:u:p:')
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
        else:
            print "invalid argument:", name
            sys.exit(2)

    if logbook_instrument is None:
        print "no instrument name found among command line parameters"
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
    # ---------------------------------------------------------

    if logbook_experiment is not None:
        if logbook_experiment == 'current':
            logbook_use_current = True
            logbook_experiment = ws_get_current_experiment ()


    # ------------------------------------------------------
    # Get a list of experiments for the specified instrument
    # and if a specific experiment was requested make sure
    # the one is in the list.
    # ------------------------------------------------------

    logbook_experiments = ws_get_experiments (logbook_experiment)

    # --------------------
    # Proceed with the GUI
    # --------------------

    root = Tk()
    root.title("LogBookGrabberUI")
    d = LogBookGrabberUI( root )
    root.mainloop()

    sys.exit(0)
