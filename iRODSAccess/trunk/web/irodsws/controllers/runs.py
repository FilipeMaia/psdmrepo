#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module runs
#------------------------------------------------------------------------

""" Pylons controller for the iRODS resources.

This software was developed for the LUSI project.  If you
use all or part of it, please give an appropriate acknowledgement.

Copyright (C) 2006 SLAC

@version $Id$ 

@author Andy Salnikov
"""

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import logging
import types

#---------------------------------
#  Imports of base class module --
#---------------------------------
from irodsws.lib.base import BaseController
import formencode

#-----------------------------
# Imports for other modules --
#-----------------------------
from pylons import request, response, session, tmpl_context as c
from pylons import config
from pylons.controllers.util import abort, redirect_to
from pylons.decorators import validate
from pylons.decorators import jsonify
from irodsws.lib.base import *
from irodsws.model.irods_model import IrodsModel

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_run_data_path = config['app_conf'].get( 'irods.run_data_path', '/psdm-zone/psdm' )

# extract run number from irods object
def _run_number ( obj ):
    if obj['type'] == 'object' :
        r = obj['name'].split('.')[0]
        r = r.split('-')
        if len(r) >= 2 and r[1][0] == 'r'  and r[1][1:].isdigit() :
            return int(r[1][1:],10)
    return None

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class RunsController(BaseController):

    @h.catch_all
    @jsonify
    def index(self, instrument, experiment, type ):
        """ GET /runs/{instrument}/{experiment}/{type} """

        # location of the experiment directory
        path = '/'.join([ _run_data_path, instrument, experiment, type ])

        # see if user can have an access
        h.checkAccess(path)

        # instantiate the model 
        model = IrodsModel()

        # make the list of runs
        runs = []
        for r in set(self._allRuns(model, path)) :
            run_url = h.url_for( action='show', 
                                instrument=instrument, 
                                experiment=experiment, 
                                type=type,
                                runs=str(r),
                                qualified=True )
            runs.append ( dict(run=r, url=run_url) )
        
        return runs
        
    @h.catch_all
    @jsonify
    def show(self, instrument, experiment, type, runs ):
        """GET /runs/{instrument}/{experiment}/{type}/{runs} """
        # url('resource', id=ID)

        # location of the experiment directory
        path = '/'.join([ _run_data_path, instrument, experiment, type ])

        # see if user can have an access
        h.checkAccess(path)

        # instantiate the model 
        model = IrodsModel()

        # make the list of runs
        allRuns = set(self._runList(runs))

        # get list of files
        files = model.files( path )
        if files is None : abort(404, "Collection does not exist: "+str(path))
        
        res = {}
        for file in files :
            
            # for file get its run number
            run = _run_number ( file )
            
            # quick check first
            if run not in allRuns : continue

            # add some additional info to the object
            name = file['collName']+'/'+file['name']
            name = name.lstrip('/')
            url = h.url_for( controller='/files', 
                             action='show', 
                             path=name,
                             qualified=True )
            file['url'] = url

            # append to the list of file for this run
            res.setdefault(run,[]).append(file)

        if not res : abort(404, "No files found for given run numbers: "+str(runs))

        # convert to the list of dictionaries
        reslist = [ dict(run=k, files=v) for k,v in res.iteritems() ]

        return reslist

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
                

    def _allRuns(self, model, path ):
        """ Generate the list of all existing runs in given directory"""

        # list all file in path
        res = model.files( path )
        if res is None : abort(404, "Collection does not exist: "+(path))

        # get file names, extract -rNNNN- part, convert it to run number
        for r in res :
            run = _run_number ( r )
            if run is not None :
                yield run
