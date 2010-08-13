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

#---------------------------------
#  Imports of base class module --
#---------------------------------
from icws.lib.base import BaseController
import formencode

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

class _CreateNewRequestForm(formencode.Schema):
    allow_extra_fields = True
    filter_extra_fields = True
    if_key_missing = ""
    instrument = formencode.validators.String(not_empty=True)
    experiment = formencode.validators.String(not_empty=True)
    runs = formencode.validators.String(not_empty=True)
    force = formencode.validators.StringBool(if_empty=False)
    priority = formencode.validators.Int(if_empty=0)

class _UpdateRequestForm(formencode.Schema):
    allow_extra_fields = True
    filter_extra_fields = True
    if_key_missing = ""
    priority = formencode.validators.Int(if_empty=0)

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class RequestController ( BaseController ) :

    #-------------------
    #  Public methods --
    #-------------------

    @jsonify
    def requests ( self, id=None ) :
        """Show requests"""

        # check user's privileges
        h.checkAccess('', '', 'read')

        model = IcdbModel()
        if id :
            try:
                id = int(id)
            except:
                abort(404)
        data = model.requests(id)
        
        # add URLs to every object
        for d in data :
            d['url'] = h.url_for( action='requests', id=d['id'] )

            if 'log' in d and d['log'] :
                try :
                    log = d['log'].split('/')
                    path = 'translator/'+log[-2]+'/'+log[-1]
                    d['log_url'] = h.url_for(controller='/log', action='show', mode='html', path=path)
                except :
                    pass

        # return list or single dict or error
        if id is None :
            return data
        elif data :
            return data[0]
        else :
            abort(404)

    @jsonify
    def create ( self ) :
        """Create new request"""
    
        # validate parameters
        schema = _CreateNewRequestForm()
        try:
            form_result = schema.to_python(dict(request.params))
        except formencode.Invalid, error:
            abort(400,str(error))
        
        instrument = form_result['instrument']
        experiment = form_result['experiment']
        runs = form_result['runs']
        force = form_result['force']
        priority = form_result['priority']

        # check user's privileges
        h.checkAccess(instrument, experiment, 'create')

        # send it all to model
        model = IcdbModel()
        res = []
        code = 200
        for run in self._runList(runs) :
            if type(run) != type(1):
                abort(400, "Invalid run number: %s (runs=%s)" % (run, runs))
            try :
                req = model.create_request(instrument, experiment, run, force, priority)
                req['url'] = h.url_for( action='requests', id=req['id'] )
                res.append(req)
            except Exception, exc:
                # failed to add request, continue with others
                #code = 400
                req = dict(instrument=instrument, experiment=experiment,
                    run=run, status='Failed', message=str(exc))
                res.append(req)

        response.status_int = code
        return res

    @jsonify
    def update ( self, id ) :
        """Create new request"""

        try:
            id = int(id)
        except Exception, e:
            abort(404, 'Non-integer request id: ' + repr(id))
    
        # check that it exists and get its info    
        model = IcdbModel()
        data = model.requests(id)
        if not data : abort(404, 'Request %d does not exist' % id)
        data = data[0]
        instr = data['instrument']
        exper = data['experiment']
        status = data['status']
        if status not in ('Initial_Entry', 'Waiting_Translation'):
            abort(405, 'Cannot update processed request: %d' % id)

        # check user's privileges
        h.checkAccess(instr, exper, 'update')
            
        # validate parameters
        schema = _UpdateRequestForm()
        try:
            form_result = schema.to_python(dict(request.params))
        except formencode.Invalid, error:
            abort(400,str(error))
        
        priority = form_result['priority']

        # send it all to model
        res = []
        code = 200
        try :
            req = model.change_request_priority(id, priority)
            req['url'] = h.url_for( action='requests', id=req['id'] )
        except Exception, exc:
            # failed to update request, continue with others
            #code = 400
            req = dict(instrument=instrument, experiment=experiment,
                run=run, status='Failed', message=str(exc))

        response.status_int = code
        return req

    @jsonify
    def delete ( self, id ) :
        """DElete request"""

        try:
            id = int(id)
        except Exception, e:
            abort(404, 'Non-integer request id: ' + repr(id))
    
        # check that it exists and get its info    
        model = IcdbModel()
        data = model.requests(id)
        if not data : abort(404, 'Request %d does not exist' % id)
        data = data[0]
        instr = data['instrument']
        exper = data['experiment']
        status = data['status']
        if status not in ('Initial_Entry', 'Waiting_Translation'):
            abort(405, 'Cannot delete processed request: %d' % id)

        # check user's privileges
        h.checkAccess(instr, exper, 'delete')

        # possible race condition here, somebody could have changed state of the request
        model.delete_request(id)
        
        return data

    @jsonify
    def experiments ( self ) :

        model = IcdbModel()
        return model.experiments()

    @jsonify
    def exp_requests ( self, instrument, experiment ) :

        # check user's privileges
        h.checkAccess(instrument, experiment, 'create')

        model = IcdbModel()
        requests = model.exp_requests(instrument, experiment)

        # add URL to every object
        for d in requests :
            d['url'] = h.url_for( action='requests', id=d['id'], instrument=None, experiment=None )

        return requests

    def _runList( self, runs ) :
        """ Generator which produces the run number list from the 
        expressions like: 1,5,7-12,54,100-128, etc. """
        
        runs = runs.split(',')
        for r in runs :
            rx = r.split('-')
            if len(rx) == 1 and rx[0].isdigit() :
                yield int(rx[0])
            elif len(rx) == 2 and rx[0].isdigit() and rx[1].isdigit() :
                rx0 = int(rx[0])
                rx1 = int(rx[1])
                for x in range(rx0,rx1+1) :
                    yield x
            else :
                # return some non-number, something unknown appears as input
                yield r
                

