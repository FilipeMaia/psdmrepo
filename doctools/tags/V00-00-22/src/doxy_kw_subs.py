#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module doxy_kw_subs...
#
#------------------------------------------------------------------------

"""Extension for sphinx which substitutes doxygen keywords.

@see RelatedModule

@version $Id$

@author Andy Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys

#---------------------------------
#  Imports of base class module --
#---------------------------------


#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------


#------------------------
# Exported definitions --
#------------------------

def setup(app):
    app.connect('autodoc-process-docstring', autodoc_process_docstring)

def autodoc_process_docstring(app, what, name, obj, options, lines):
    for i in range(len(lines)):
        if '@' in lines[i]:
            lines[i] = lines[i].replace('@see ', ':emphasis:`See also:` ')
            lines[i] = lines[i].replace('@version ', ':emphasis:`Version:` ')
            lines[i] = lines[i].replace('@author ', ':emphasis:`Author:` ')


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
