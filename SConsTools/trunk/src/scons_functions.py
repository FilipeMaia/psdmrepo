#===============================================================================
#
# Main SCons script for LUSI release building
#
# $Id$
#
#===============================================================================

import os
import sys
from os.path import join as pjoin

from trace import *

# ==========================
#     Local functions
# ==========================

def fail ( msg, code = 2 ):
    print >> sys.stderr, msg
    print >> sys.stderr, "Exiting with code %d" % code
    Exit(code)

#
# Create directory or fail gracefully
#
def mkdirOrFail ( d ) :
    try :
        if not os.path.isdir( d ) :
            os.makedirs ( d )
            trace ( "Creating directory `%s'" % d, "mkdirOrFail", 1 )
    except :
        fail ( "Failed to create `%s' directory" % ( d, ) )

#
# Create the links in the release directory to package directories, e.g.:
#
#  TOP/include/Package -> ../Package/include 
#
def makePackageLinks ( dirname, packagelist ) :
    
    packageset = set(packagelist)
    mkdirOrFail ( dirname )
    existing = set ( os.listdir ( dirname ) )
    
    # links to delete
    extra = existing - packageset
    for f in extra :
        fname = pjoin( dirname, f )
        try :
            os.remove ( fname )
            trace ( "Removing link `%s'" % fname, "pkgLinks", 1 )
        except :
            fail ( "Failed to remove `%s'" % ( fname, ) )
            
    # links to create
    missing = packageset - existing
    for f in missing :
        if os.path.isdir ( pjoin( f, dirname ) ) :
            src = pjoin( "..", f, dirname )
            dst = pjoin( dirname, f )
            try :
                os.symlink ( src, dst )
                trace ( "Creating link `%s'" % dst, "pkgLinks", 1 )
            except :
                fail ( "Failed to create symlink `%s'" % ( dst, ) )

