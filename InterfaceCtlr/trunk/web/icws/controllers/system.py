#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module system...
#
#------------------------------------------------------------------------

"""Pylons controller for the Inteface Controlles system-level resource.

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
import simplejson

#---------------------------------
#  Imports of base class module --
#---------------------------------
from icws.lib.base import BaseController

#-----------------------------
# Imports for other modules --
#-----------------------------
from pylons.decorators import jsonify
from pylons.controllers.util import abort
from icws.lib.base import *
from icws.model.icdb_model import IcdbModel

#----------------------------------
# Local non-exported definitions --
#----------------------------------
def _quote_attr(val):
    if not val: return '""'
    val = str(val)
    if '"' not in val: return '"'+val+'"'
    if "'" not in val: return "'"+val+"'"
    return '"'+val.replace('"', '&quot;')+'"'

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class SystemController ( BaseController ) :

    #-------------------
    #  Public methods --
    #-------------------

    def status ( self, id = None, renderer="json" ) :

        # check user's privileges
        h.checkAccess('', '', 'read')

        model = IcdbModel()
        cstatus = model.controller_status(id)
        for r in cstatus :
            if 'log' in r and r['log'] :
                try :
                    # add URL for the log
                    log = r['log'].split('/')
                    path = 'system/'+log[-2]+'/'+log[-1]
                    r['log_url'] = h.url_for(controller='/log', action='show', mode='html', path=path, qualified=True)
                except :
                    pass
        if id is not None and not cstatus:
            # no such ID
            abort(404)
            
        if renderer == 'xml':
            response.content_type = 'application/xml'
            res = ['<?xml version="1.0" encoding="UTF-8"?>\n', '<controllers>\n']
            for ctrl in cstatus:
                ctrl['instruments'] = ','.join(ctrl['instruments'])
                ctrl['stop_url'] = h.url_for(controller='system', action='stop', id=ctrl['id'], qualified=True)
                cstr = ["<controller"]
                for k, v in ctrl.items():
                    cstr.append('%s=%s' % (k, _quote_attr(v)))
                res.append(' '.join(cstr) +'/>')
            res += ['</controllers>\n']
            return res
        else:
            if id is not None:
                # return single object
                cstatus = cstatus[0]
            response.content_type = 'application/json'
            return simplejson.dumps(cstatus)


    @jsonify
    def stop ( self, id ) :

        # check user's privileges
        h.checkAccess('', '', 'delete')

        model = IcdbModel()
        res = model.controller_stop(id)
        if res : return res
        # no such request
        abort(404)
