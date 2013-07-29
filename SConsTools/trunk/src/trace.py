#===============================================================================
#
# Main SCons script for SIT release building
#
# $Id$
#
#===============================================================================

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

trace_level = 0

def setTraceLevel ( level ):
    global trace_level 
    trace_level = level
#
# Print debug message
#
def trace ( msg, src, level ) :
    #print "trace_level=", trace_level, "level=", level
    if trace_level >= level :
        logging.info( "{%s-%d}  %s" % ( src, level, msg) )
