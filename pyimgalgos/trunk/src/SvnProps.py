#!/usr/bin/env python
#------------------------------
"""
Class SvnProps retreives/stores/provide access to svn properties.

NOTE: In order to always update this file when revision changes, use command:
svn propset svn:keywords "Revision" SvnProps.py
Also see: ~/.subversion/config

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhai S. Dubrovin
"""

#------------------------------
# Properties substituted by SVN 
#------------------------------
# __revision__ = "$Revision$"
# __author__   = "$Author$"
#------------------------------

import sys

#------------------------------

class SvnProps :

    revision = "$Revision$"
    author   = "$Author$"
    id       = "$Id$"
    headurl  = "$HeadURL$"
    header   = "$Header:$"
    datelc   = "$LastChangedDate$"
    date     = "$Date$"

#------------------------------

    def __init__(self) :
        self.str_revnum = self.get_prop_value(self.revision) # ex: '8029' 
        self.str_url    = self.get_prop_value(self.headurl)  # ex: 'https://pswww.slac.stanford.edu/svn/...' 

#------------------------------
        
    def get_prop_value(self, str_prop) :
        """Parses thes string of property and returns the value"""
        return str_prop.split(':')[1].rstrip('$').strip()

#------------------------------
        
    def get_pkg_revision(self) :
        """Returns the module revision number"""
        return self.str_revnum

#------------------------------

    def get_pkg_name_from_path(self) :
        """Returns the package name"""
        path = __file__
        fields = path.split('/')
        return 'N/A' if len(fields)<3 else fields[-3] 

#------------------------------

    def get_pkg_name(self) :
        """Returns the package name"""
        fields = self.str_url.split('/')
        return 'N/A' if len(fields)<4 else fields[-4] 
        #return self.get_pkg_name_from_path() if len(fields)<4 else fields[-4] 

#------------------------------

svnprops = SvnProps()  # use it as singleton

#------------------------------

if __name__ == "__main__" :
    
    msg  = '__file__   : %s' % __file__
    msg += '\n__name__   : %s' % __name__
    msg += '\nsys.argv[0]: %s' % sys.argv[0]
    print msg

    print 'svnprops.get_pkg_revision():', svnprops.get_pkg_revision()
    print 'svnprops.get_pkg_name()    :', svnprops.get_pkg_name()

    sys.exit ( 'End of test' )

#------------------------------
