#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Detector...
#
#------------------------------------------------------------------------

"""Brief one-line description of the module.

Following paragraphs provide detailed description of the module, its
contents and usage. This is a template module (or module template:)
which will be used by programmers to create new Python modules.
This is the "library module" as opposed to executable module. Library
modules provide class definitions or function definitions, but these
scripts cannot be run by themselves.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhail S. Dubrovin
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

##-----------------------------
import sys



##-----------------------------

class Detector :
    """Brief description of a class.

    Full description of this class.

    @see BaseClass
    @see OtherClass
    """

    def __init__ (self, source, pbits) :
        """Constructor.
        @param source   first parameter
        @param y   second parameter
        """

        self.source = source
        self.pbits = pbits
        #self.dettype = ImgAlgos::detectorTypeForSource(m_source);
        #self.cgroup  = ImgAlgos::calibGroupForDetType(m_dettype); // for ex: "PNCCD::CalibV1";


##-----------------------------

#from psana import *
import psana

if __name__ == "__main__" :

    str_ds = 'exp=cxif5315:run=169'
    ds = psana.DataSource(str_ds)

    env = ds.env()
    cls = env.calibStore()
    eviter = ds.events()
    evt = eviter.next()

    for key in evt.keys() : print key

    src = psana.Source('DetInfo(CxiDs2.0:Cspad.0)')

    d = evt.get(psana.CsPad.DataV2, src)
    print 'd.TypeId: ', d.TypeId

    q0 = d.quads(0)
    q0_data = q0.data()
    print 'q0_data.shape: ', q0_data.shape

    sys.exit ( "Module is not supposed to be run as main module" )

##-----------------------------
