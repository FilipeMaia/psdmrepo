#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module TemplateLoader...
#
#------------------------------------------------------------------------

"""Implementation of a Jinja template loader.

This loader looks for files in $SIT_DATA/psddl/templates directories.
One template file can contain multiple templates, each template is
preceeded by a line starting with "::::template::::" string followed
by spaces and the name of a template which is a single word, any words 
following template name are ignored (can be used for comments). Lines
starting with colon are ignored.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

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
import os
import re

#---------------------------------
#  Imports of base class module --
#---------------------------------
import jinja2 as ji

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------


#---------------------
#  Class definition --
#---------------------
class TemplateLoader(ji.FileSystemLoader):

    #----------------
    #  Constructor --
    #----------------
    def __init__(self, package='psddl', templateSubDir='templates'):
        '''Loads templates for psddl. Similar to the jinja2 PackageLoader()
        function but adapted for packages within the SIT build system
        (as opposed to Python packages.)
        ARGS:
        templateSubdir -  a subdirectory to the data directory of the 
        package.
        '''
        self.package=package
        self.templateSubDir=templateSubDir
        path = os.environ['SIT_DATA'].split(':')
        ji.FileSystemLoader.__init__(self, path)


    #-------------------
    #  Public methods --
    #-------------------

    def get_source(self, environment, template):

        # template name is the file name followed by "?template"
        fname, template = template.split('?')

        # prepend package/templateSubDir to path (defaults to psddl/templates)
        fname = os.path.join("%s/%s" % (self.package,self.templateSubDir), fname)

        # call base class method
        source, path, helper = ji.FileSystemLoader.get_source(self, environment, fname)

        # iterate over lines, find a template
        tmpl = []
        collect = False
        for line in source.splitlines(True):
            words = line.split()
            if words and words[0] == "::::template::::":
                if words[1] == template:
                    collect = True
                else:
                    collect = False
            elif line and line[0] == ':':
                pass
            elif collect: 
                tmpl.append(line)
                
        return ''.join(tmpl), path, helper

    #--------------------------------
    #  Static/class public methods --
    #--------------------------------

    #--------------------
    #  Private methods --
    #--------------------

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
