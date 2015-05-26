#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module queue
#------------------------------------------------------------------------

""" Pylons controller for the iRODS queue.

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

#---------------------------------
#  Imports of base class module --
#---------------------------------
from irodsws.lib.base import BaseController

#-----------------------------
# Imports for other modules --
#-----------------------------
from pylons import request, response, session, tmpl_context as c
from pylons.controllers.util import abort, redirect_to
from pylons.decorators import validate
from pylons.decorators import jsonify
from irodsws.lib.base import *
from irodsws.model.irods_model import IrodsModel

#----------------------------------
# Local non-exported definitions --
#----------------------------------

_log = logging.getLogger(__name__)

class QueueController(BaseController):

    @h.catch_all
    @jsonify
    def index(self):
        """GET /queue: Return all rules in a queue"""
        
        model = IrodsModel()
        res = model.queue()
        return res

