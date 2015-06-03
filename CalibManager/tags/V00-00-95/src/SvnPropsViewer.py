#!/usr/bin/env python
#------------------------------
"""
Class SvnProps retreives and provides access to svn properties.

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

$Revision$

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

class SvnPropsViewer :

#------------------------------

    def __init__ (self, props) :
        self.props    = props

        self.revision = props.revision
        self.author   = props.author  
        self.id       = props.id      
        self.headurl  = props.headurl 
        self.header   = props.header  
        self.datelc   = props.datelc  
        self.date     = props.date    

        self.str_revnum = self.get_prop_value(self.revision) # ex: '8029' 
        self.str_url    = self.get_prop_value(self.headurl)  # ex: 'https://pswww.slac.stanford.edu/svn/...' 

#------------------------------
        
    def get_prop_value(self, str_prop) :
        """Parses thes string of property and returns the value"""
        return str_prop.split(':',1)[1].rstrip('$').strip()

#------------------------------
        
    def get_pkg_revision(self) :
        """Returns the package revision number"""
        return self.str_revnum

#------------------------------

    def get_pkg_name(self) :
        """Returns the package name"""
        fields = self.str_url.split('/')
        return 'N/A' if len(fields)<4 else fields[-4] 
        #return self.get_pkg_name_from_path() if len(fields)<4 else fields[-4] 

#------------------------------

# FOR TEST PURPOSE ONLY:
from CalibManager.SvnProps   import svnprops as spcm
#from ImgAlgos.SvnProps       import svnprops as spia
#from PSCalib.SvnProps        import svnprops as spps
#from pdscalibdata.SvnProps   import svnprops as spcd
#from CSPadPixCoords.SvnProps import svnprops as sppc

#------------------------------

if __name__ == "__main__" :

    props = spcm
    pview = SvnPropsViewer(props)
    
    print 'pview.get_pkg_revision()       :', pview.get_pkg_revision()
    print 'pview.get_pkg_name()           :', pview.get_pkg_name()
    print 'pview.get_prop_value(props.id) :', pview.get_prop_value(props.id)

    sys.exit ( 'End of test' )

#------------------------------
