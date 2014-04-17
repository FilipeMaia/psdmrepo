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


    def get_version(self) :
        try :
            return gu.get_pkg_version('CalibManager') # Very slow
            #return cp.package_versions.get_pkg_version('CalibManager')
        except :
            return 'N/A'

#------------------------------

if __name__ == "__main__" :

    ndb = NotificationDBForCL()
    main_test(ndb)
    ndb.close()

    sys.exit ( 'End of test NotificationDBForCL' )

#------------------------------
