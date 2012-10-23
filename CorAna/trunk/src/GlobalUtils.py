#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module GlobalUtils...
#
#------------------------------------------------------------------------

"""Contains Global Utilities

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: template!python!py 4 2008-10-08 19:27:36Z salnikov $

@author Mikhail S. Dubrovin
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
#import time
from time import localtime, gmtime, strftime, clock, time

#-----------------------------
# Imports for other modules --
#-----------------------------
#from PkgPackage.PkgModule import PkgClass

#import ConfigParameters as cp

#------------------------
# Exported definitions --
#------------------------

#----------------------------------
#----------------------------------
#----------------------------------
#----------------------------------

def get_item_last_name(dsname):
    """Returns the last part of the full item name (after last slash)"""

    path,name = os.path.split(str(dsname))
    return name

def get_item_path_to_last_name(dsname):
    """Returns the path to the last part of the item name"""

    path,name = os.path.split(str(dsname))
    return path

def get_item_path_and_last_name(dsname):
    """Returns the path and last part of the full item name"""

    path,name = os.path.split(str(dsname))
    return path, name

#----------------------------------

def get_item_second_to_last_name(dsname):
    """Returns the 2nd to last part of the full item name"""

    path1,name1 = os.path.split(str(dsname))
    path2,name2 = os.path.split(str(path1))

    return name2 

#----------------------------------

def get_item_third_to_last_name(dsname):
    """Returns the 3nd to last part of the full item name"""

    path1,name1 = os.path.split(str(dsname))
    path2,name2 = os.path.split(str(path1))
    path3,name3 = os.path.split(str(path2))

    str(name3)

    return name3 

#----------------------------------

def get_item_name_for_title(dsname):
    """Returns the last 3 parts of the full item name (after last slashes)"""

    path1,name1 = os.path.split(str(dsname))
    path2,name2 = os.path.split(str(path1))
    path3,name3 = os.path.split(str(path2))

    return name3 + '/' + name2 + '/' + name1

#----------------------------------

def get_time_sec():
    return time()

#----------------------------------

#def get_time_sec():
#    return clock()

#----------------------------------

def get_current_local_time_tuple():
    return localtime()

def get_current_gm_time_tuple():
    return gmtime()

#----------------------------------

def get_current_local_time_stamp(fmt='%Y-%m-%d %H:%M:%S %Z'):
    return strftime(fmt, localtime())

def get_current_gm_time_stamp(fmt='%Y-%m-%d %H:%M:%S %Z'):
    return strftime(fmt, gmtime())
    
#----------------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    print 'Time (sec) :', int( get_time_sec() )

    print 'Time local :', get_current_local_time_tuple()
    print 'Time (GMT) :', get_current_gm_time_tuple()

    print 'Time local :', get_current_local_time_stamp()
    print 'Time (GMT) :', get_current_gm_time_stamp()

    sys.exit ( "End of test" )

#----------------------------------
