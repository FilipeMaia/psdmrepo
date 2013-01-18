#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module request...
#
#------------------------------------------------------------------------

"""Pylons controller for the Interface Controller request-level resource.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Andrei Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os

#---------------------------------
#  Imports of base class module --
#---------------------------------
from icws.lib.base import BaseController
import formencode

#-----------------------------
# Imports for other modules --
#-----------------------------
from pylons import config
from pylons.controllers.util import abort
from icws.lib.base import *
from icws.model.icdb_model import IcdbModel

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class LogController ( BaseController ) :

    #-------------------
    #  Public methods --
    #-------------------

    def show ( self, mode, path ) :
        """get log file"""

        # check user's privileges
        h.checkAccess('', '', 'read')

        app_conf = config['app_conf']

        path = path.split('/')        
        if path[0] == 'system' :
            logdir = app_conf.get( 'log.sys_logdir', '' )
            log = os.path.join(logdir, '/'.join(path[1:]))
        elif path[0] == 'translator' :
            logdir = app_conf.get( 'log.tran_logdir', '' )
            log = os.path.join(logdir, '/'.join(path[1:]))
        else :
            logdir = app_conf.get( 'log.other_logdir', '' )
            log = os.path.join(logdir, '/'.join(path))
            
        # try to open the file
        try :
            f = open(log, 'rU')
        except IOError, e:
            #return "Cannot open file "+log
            abort(404, unicode(e))

        if mode == 'text' :
    
            # dump it as text
            res = f.read()
            f.close()
            
            response.content_type = 'text/plain'
            return res

        elif mode == 'html' :

            # very simple error highlighting algorithm
            
            res = [ '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">\n' ]
            res += [ '<html><title>Log file ' + '/'.join(path) + '</title><body><pre>\n' ]
            res += [ '<span style="color:black">' ]
            
            color = 'black'
            for line in f :
                new_color = color
                if line.startswith('[ERR]') :
                    new_color = 'red'
                elif line.startswith('[WRN]') :
                    new_color = '#cc8800'
                elif line.startswith('[INF]') :
                    new_color = 'black'
                elif line.startswith('[TRC]') :
                    new_color = 'black'
                elif line.startswith('[DBG]') :
                    new_color = 'black'
                if new_color != color :
                    color = new_color
                    res += [ '</span><span style="color:%s">' % color ]

                res.append(line)
                
            res += [ '</span></pre>\n' ]
            res += [ '</body></html>\n' ]
            
            return ''.join(res)
        
        else :
            
            abort(404, "Unknown format type: "+mode)
