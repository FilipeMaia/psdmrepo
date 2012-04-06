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
import tempfile

import tkMessageBox
from Tkinter import *
from ScrolledText import *

import urllib
import urllib2

##############################################
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
        self.message = ScrolledText(parent, width=48,height=9)
        self.message.grid(row=1,column=1,columnspan=4,rowspan=4,padx=5,pady=5,sticky=W+N)

        Label(parent,text="Experiment:").grid(row=1,column=5,padx=5,pady=5,sticky=W+N)
        if logbook_experiment is None:
            self.experiment = Listbox(parent,height=5)
            for e in logbook_experiments.keys():
                self.experiment.insert(END, e)
            self.experiment.select_set(0)
            self.experiment.grid(row=1,column=6,rowspan=1,padx=5,pady=5,sticky=W+N)
            self.experiment.bind("<<ListboxSelect>>", self.onSelectExperiment)
        else:
            self.exp_name = StringVar()
            self.exp_name.set(logbook_experiment)
            Label(parent,textvariable=self.exp_name).grid(row=1,column=6,padx=5,pady=5,sticky=W+N)

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

        Label(parent, text="Tag: ").grid(row=10,column=0,padx=5,pady=5,sticky=W+N)
        tags4experiment = []
        if logbook_experiment is None:
            for e in logbook_experiments.keys():
                tags4experiment = logbook_experiments[e]['tags']
                break
        else:
            tags4experiment = logbook_experiments[logbook_experiment]['tags']

        self.tags = Listbox(parent,height=5)
        self.tags.insert(END,"SCREENSHOT")
        for t in tags4experiment:
            if t != "SCREENSHOT": self.tags.insert(END, t)
        self.tags.select_set(0)
        self.tags.grid(row=10,column=1,rowspan=1,padx=5,pady=5,sticky=W+N)

        Label(parent, text="<- scroll to select an existing tag").grid(row=10,column=2,columnspan=4,padx=5,pady=5,sticky=W+N)
        self.tag = Entry(parent)
        self.tag.insert(0, "")
        self.tag.grid(row=11,column=1,columnspan=2,padx=5,pady=0,sticky=W+N)
        Label(parent, text="<- edit this tag if needed").grid(row=11,column=2,columnspan=4,padx=5,pady=0,sticky=W+N)

        self.tags.bind("<<ListboxSelect>>", self.onSelectTag)

        self.canvas = Canvas(background='black',width=0,height=0)
        self.canvas.grid(row=15,column=0,padx=5,pady=10,columnspan=6,sticky=W+N+E+S)

        self.submit.configure(state=DISABLED)
        self.cancel.configure(state=DISABLED)

    def onSelectExperiment(self,val):
        self.tags.delete(0,END)
        self.tags.insert(END,"SCREENSHOT")
        for t in logbook_experiments[self.experiment.get(self.experiment.curselection())]['tags']:
            if t != "SCREENSHOT": self.tags.insert(END, t)
        self.tags.select_set(0)
        
    def onSelectTag(self,val):
        self.tag.delete(0,END)
        self.tag.insert(0,self.tags.get(self.tags.curselection()))

    def on_grab(self):

        self.f_jpeg = tempfile.NamedTemporaryFile(mode='r+b',suffix='.jpeg')
        f_gif = tempfile.NamedTemporaryFile(mode='r+b',suffix='.gif')
        print self.f_jpeg.name, f_gif.name
        if( 0 == os.system("import -trim -frame -border %s; convert %s %s" % (self.f_jpeg.name, self.f_jpeg.name, f_gif.name))):
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

        tag_name = self.tag.get()

        exper_id = logbook_experiments[exper_name]['id']

        url = ws_url+'/LogBook/NewFFEntry4grabberJSON.php'
        run = self.run.get()
        message_id = self.message_id.get()

        child_output = ''
        if logbook_child_cmd is not None: child_output = os.popen(logbook_child_cmd).read()

        if(( run != '' ) and ( message_id != '' )):
            tkMessageBox.showerror (
                "Inconsistent Input",
                "Run number can't be used togher with the parent message ID. Choose the right context to post the screenshot and try again." )
            return
        if( run != '' ):
            datagen,headers = multipart_encode([
                ('author_account' , logbook_author),
                ('id'             , exper_id),
                ('message_text'   , self.message.get("0.0",END)),
                ('scope'          , 'run'),
                ('run_num'        , run),
                ('num_tags'       , '1'),
                ('tag_name_0'     , tag_name),
                ('tag_value_0'    , ''),
                ('text4child'     , child_output),
                MultipartParam.from_file("file1", self.f_jpeg.name),
                ("file1"          , self.descr.get()) ])
        elif( message_id != '' ): 
            datagen,headers = multipart_encode([
                ('author_account' , logbook_author),
                ('id'             , exper_id),
                ('message_text'   , self.message.get("0.0",END)),
                ('scope'          , 'message'),
                ('message_id'     , message_id),
                ('num_tags'       , '1'),
                ('tag_name_0'     , tag_name),
                ('tag_value_0'    , ''),
                ('text4child'     , child_output),
                MultipartParam.from_file("file1", self.f_jpeg.name),
                ("file1"          , self.descr.get()) ])
        else:
            datagen,headers = multipart_encode([
                ('author_account' , logbook_author),
                ('id'             , exper_id),
                ('message_text'   , self.message.get("0.0",END)),
                ('scope'          , 'experiment'),
                ('num_tags'       , '1'),
                ('tag_name_0'     , tag_name),
                ('tag_value_0'    , ''),
                ('text4child'     , child_output),
                MultipartParam.from_file("file1", self.f_jpeg.name),
                ("file1"          , self.descr.get()) ])

        try:
            req = urllib2.Request(url, datagen, headers)
            response = urllib2.urlopen(req)
            the_page = response.read()
            result = simplejson.loads(the_page)
            if result['status'] != 'success':
                tkMessageBox.showerror ( "Error", result['message'])
            print 'New message ID:', int(result['message_id'])

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
                        d[experiment]['tags'] = ws_get_tags(e['id'])
            else:
                for e in result['ResultSet']['Result']:
                    d[e['name']] = e
                    d[e['name']]['tags'] = ws_get_tags(e['id'])

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

def ws_get_tags (id):

    url = ws_url+'/LogBook/RequestUsedTagsAndAuthors.php?id='+id;

    try:
        req = urllib2.Request(url)
        response = urllib2.urlopen(req)
        the_page = response.read()
        result = simplejson.loads(the_page)
        if result['Status'] != 'success':
            print "ERROR: failed to obtain tags for experiment id=%d because of:" % id,result['Message']
            sys.exit(1)
        return result['Tags']

    except urllib2.URLError, reason:
        print "ERROR: failed to get a list of tags for experiment id=%d from Web Service due to: " % id, reason
        sys.exit(1)
    except urllib2.HTTPError, code:
        print "ERROR: failed to get a list of tags for experiment id=%d from Web Service due to: " % id, code
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

    -c command-for-child-message

      allows posting a child message with a standard output of the specified
      UNIX command (or an application) found after the option. The child
      will be posted immediattely after the parrent message.

      NOTE: using this option may delay the time taken by the Grabber
            by a duration of time needed by the specified command to produce
            its output

""" % (progname)

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:e:w:u:p:c:')
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
        elif name in ('-c',):
            logbook_child_cmd = value
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
