#!/usr/bin/env python
#------------------------------
"""
Class SvnProps stores updated by svn properties.

NOTE: In order to always update revision number in this file when revision changes, use command:
svn propset svn:keywords "Revision" SvnProps.py
Also see: ~/.subversion/config

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

@version $Id: SvnProps.py 8029 2014-04-18 17:16:45Z dubrovin@SLAC.STANFORD.EDU $

@author Mikhai S. Dubrovin
"""

#------------------------------
# Properties substituted by SVN 
#------------------------------
# __revision__ = "$Revision$"
# __author__   = "$Author: dubrovin@SLAC.STANFORD.EDU $"
#------------------------------

import sys

#------------------------------

class SvnProps :

    updated  = "2014-04-25"
    revision = "$Revision$"
    author   = "$Author: dubrovin@SLAC.STANFORD.EDU $"
    id       = "$Id: SvnProps.py 8029 2014-04-18 17:16:45Z dubrovin@SLAC.STANFORD.EDU $"
    headurl  = "$HeadURL: https://pswww.slac.stanford.edu/svn/psdmrepo/CalibManager/trunk/src/SvnProps.py $"
    header   = "$Header:$"
    datelc   = "$LastChangedDate: 2014-04-18 10:16:45 -0700 (Fri, 18 Apr 2014) $"
    date     = "$Date: 2014-04-18 10:16:45 -0700 (Fri, 18 Apr 2014) $"

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
