#===============================================================================
#
# Main SCons script for SIT release building
#
# $Id$
#
#===============================================================================

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
        print msg, ( "{%s-%d}" % ( src, level ) )

