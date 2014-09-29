#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module active...
#
#------------------------------------------------------------------------

"""Pylons controller for the Interface Controller's active experiments list.

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

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class ActiveController ( BaseController ) :

    #-------------------
    #  Public methods --
    #-------------------

    @jsonify
    def index ( self, instrument=None, experiment=None ) :
        """Show requests"""

        # check user's privileges
        h.checkAccess('', '', 'read')

        model = IcdbModel()
        data = model.active_index(instrument, experiment)
        
        # add URL to every object
        for d in data :
            d['url'] = h.url_for( action='index', instrument=d['instrument'], experiment=d['experiment'])

        # return list or single dict or error
        if instrument is None and experiment is None:
            return data
        elif data :
            return data[0]
        else :
            abort(404)

    @jsonify
    def add ( self, instrument, experiment) :
        """Create new active experiment"""
    
        # check user's privileges
        h.checkAccess(instrument, experiment, 'create')

        # send it all to model
        model = IcdbModel()
        return model.activate_experiment(instrument, experiment)

    @jsonify
    def delete ( self, instrument, experiment ) :
        """Create new request"""

        # check user's privileges
        h.checkAccess(instrument, experiment, 'delete')

        # send it all to model
        model = IcdbModel()
        return model.deactivate_experiment(instrument, experiment)

