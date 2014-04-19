#!/usr/bin/env python
#------------------------------
"""
Class SvnProps stores updated by svn properties.

NOTE: In order to always update revision number in this file when revision changes, use command:
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

svnprops = SvnProps()  # use it as singleton

#------------------------------

if __name__ == "__main__" :
    
    print 'svnprops.revision : %s' % svnprops.revision
    print 'svnprops.author   : %s' % svnprops.author  
    print 'svnprops.id       : %s' % svnprops.id      
    print 'svnprops.headurl  : %s' % svnprops.headurl 
    print 'svnprops.header   : %s' % svnprops.header  
    print 'svnprops.datelc   : %s' % svnprops.datelc  
    print 'svnprops.date     : %s' % svnprops.date    

    sys.exit ( 'End of test' )

#------------------------------
