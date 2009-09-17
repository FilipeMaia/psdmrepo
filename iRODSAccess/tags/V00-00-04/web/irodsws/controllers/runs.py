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

_run_data_path = config['app_conf'].get( 'irods.run_data_path', '/psdm-zone/exp' )

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class RunsController(BaseController):

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
        for r in self._allRuns(model, path) :
            run_url = h.url_for( action='show', 
                                instrument=instrument, 
                                experiment=experiment, 
                                type=type,
                                runs=str(r),
                                qualified=True )
            runs.append ( dict(run=r, url=run_url) )
        
        return runs
        
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

        # make the list of all runs
        allRuns = set([ r for r in self._allRuns(model, path) ])
        
        reslist = []
        for run in self._runList( runs ) :
            # for every run get the list of all files in that run's directory
            
            if not isinstance(run,types.IntType) : 
                abort(404,"bad run number specified: "+str(run) )

            # quick check first
            if run not in allRuns : continue
            
            # get file list recursively
            runpath = "%s/%06d" % ( path, run )
            res = model.files( runpath, recursive = True )
            
            # filter out collections, only report files
            files = []
            if res is not None :
                for r in res :
                    if r['type'] == 'object' :
                        name = r['collName']+'/'+r['name']
                        name = name.lstrip('/')
                        url = h.url_for( controller='/files', 
                                         action='show', 
                                         path=name,
                                         qualified=True )
                        r['url'] = url
                        files.append(r)
                        
            reslist.append( dict(run=run, files=files) )

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
        if res is None : abort(404)

        # extract the sub-directory names and convert them to numbers
        for r in res :
            if r['type'] == 'collection' :
                r = r['name'].split('/')[-1]
                if r.isdigit() : 
                    yield int(r)

    