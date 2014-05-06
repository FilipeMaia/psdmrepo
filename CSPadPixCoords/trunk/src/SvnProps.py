#!/usr/bin/env python
#------------------------------
"""
Class SvnProps stores updated by svn properties.

NOTE: To update revision number in this file when revision changes, use command:
psvn mktxtprop src/SvnProps.py
or
svn propset svn:keywords "Revision" src/SvnProps.py
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
    def __init__(self) : 
        self.updated  = "2014-05-05"
        self.revision = "$Revision$"
        self.author   = "$Author$"
        self.id       = "$Id$"
        self.headurl  = "$HeadURL: https://pswww.slac.stanford.edu/svn/psdmrepo/CSPadPixCoords/trunk/src/SvnProps.py $"
        self.header   = "$Header:$"
        self.datelc   = "$LastChangedDate$"
        self.date     = "$Date$"

#------------------------------

svnprops = SvnProps()  # use it as a singleton

#------------------------------

if __name__ == "__main__" :
    
    print 'svnprops.updated  : %s' % svnprops.updated
    print 'svnprops.revision : %s' % svnprops.revision
    print 'svnprops.author   : %s' % svnprops.author
    print 'svnprops.id       : %s' % svnprops.id
    print 'svnprops.headurl  : %s' % svnprops.headurl
    print 'svnprops.header   : %s' % svnprops.header
    print 'svnprops.datelc   : %s' % svnprops.datelc
    print 'svnprops.date     : %s' % svnprops.date

    sys.exit ( 'End of test' )

#------------------------------
