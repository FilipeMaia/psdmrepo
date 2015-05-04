#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module NotificationDBForCL.py...
#
#------------------------------------------------------------------------

"""
This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see 

@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------

from NotificationDB import *

#------------------------------

class NotificationDBForCL (NotificationDB):
    """Is intended for submission of notification records in db
    """
    def __init__(self) :
        NotificationDB.__init__(self, table='calibrun')


#------------------------------

if __name__ == "__main__" :

    ndb = NotificationDBForCL()
    main_test(ndb)
    ndb.close()

    sys.exit ( 'End of test NotificationDBForCL' )

#------------------------------
