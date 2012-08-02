#===============================================================================
#
# Main SCons script for SIT release building
#
# $Id$
#
#===============================================================================

import os
import sys
from os.path import join as pjoin

from trace import *

from SCons.Defaults import *
from SCons.Script import *

# ==========================
#     Local functions
# ==========================

def warning(msg):
    print >> sys.stderr, "WARNING:", msg

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
            trace ( "Creating directory (1) `%s'" % d, "mkdirOrFail", 1 )
    except :
        fail ( "Failed to create `%s' directory" % ( d, ) )

#
# Create the links in the release directory to package directories, e.g.:
#
#  TOP/include/Package -> ../Package/include 
#
def makePackageLinks ( dirname, packagelist ) :
    
    packageset = set(packagelist)
    try:
        existing = set(os.listdir(dirname))
    except OSError:
        # likely does not exist
        existing = set()
    
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
            mkdirOrFail ( dirname )
            src = pjoin( "..", f, dirname )
            dst = pjoin( dirname, f )
            try :
                os.symlink ( src, dst )
                trace ( "Creating link `%s'" % dst, "pkgLinks", 1 )
            except :
                fail ( "Failed to create symlink `%s'" % ( dst, ) )


def MyGlob(pattern, ondisk=True, source=True, strings=False, recursive=False):
    """ Recursive glob function """
    
    dirname = os.path.dirname(pattern)
    pattern = os.path.basename(pattern)
    
    trace ( "dirname=%s pattern=%s" % (dirname, pattern), "MyGlob", 7 )
    names = Glob(os.path.join(dirname, pattern), ondisk, source, strings)

    if recursive :
        
        for entry in Glob(os.path.join(dirname, '*'), source=True, strings=False):
            
            #trace ( "entry=`%s' %s class=%s" % (entry.get_abspath(), repr(entry), entry.__class__), "MyGlob", 7 )
            if entry.__class__ is SCons.Node.FS.Dir :
                names += MyGlob(os.path.join(str(entry), pattern), ondisk, source, strings, recursive)
                
    return names
