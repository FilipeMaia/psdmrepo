#!/usr/bin/env python
"""
Class SvnProps retreives/stores/provide access to svn properties.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhail S. Dubrovin
"""

#--------------------------------
# Properties substituted by SVN -
#--------------------------------
__revision__ = "$Revision$"
__author__   = "$Author$"
__author__   = "$Date$"
__package__  = "$Package: CalibManager$"
#--------------------------------

import sys

#--------------------------------

class SvnProps :

    revision = "$Revision$"
    author   = "$Author$"
    id       = "$Id$"
    headurl  = "$HeadURL$"
    header   = "$Header:$"
    datelc   = "$LastChangedDate$"
    date     = "$Date$"

    def __init__(self) :
        self.str_revision = self.revision.split(':')[1].rstrip('$').strip()
        
    def get_pkg_revision(self) :
        """Returns the package revision number"""
        return self.str_revision

#--------------------------------

    def get_pkg_name(self) :
        """Returns the package name"""
        path = __file__
        fields = path.split('/')
        return 'N/A' if len(fields)<3 else fields[-3] 

#------------------------------

if __name__ == "__main__" :
    
    svnprops = SvnProps()     
    msg  = '__file__   : %s' % __file__
    msg += '\n__name__   : %s' % __name__
    msg += '\nsys.argv[0]: %s' % sys.argv[0]
    print msg

    print 'svnprops.get_pkg_revision():', svnprops.get_pkg_revision()
    print 'svnprops.get_pkg_name()    :', svnprops.get_pkg_name()

    sys.exit ( 'End of test' )

#------------------------------
