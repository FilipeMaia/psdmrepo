#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: codegen_kwgen.py 4 2008-10-08 19:27:36Z salnikov$
#
# Description:
#  Python script which is used by `codegen' script to provide some
#  language-specific operations.
#
#------------------------------------------------------------------------

#
# Typically languages use some specific code for handling 
# base class declaration/initialization which cannot be put into
# the main codegen script. We define several hooks here to handle
# these operations. For languages which support base classes one 
# has to define here three methods:
#
#    codegen_<lang>_BASEDECL  - for base class declaration
#    codegen_<lang>_BASEINIT  - for base class initialization
#    codegen_<lang>_BASEINCL  - for base class "include" (or import)
#
# These methods should return pieces of code (strings) which will replace
# corresponding tokens in the template files. if they return None then empty
# string will be used instead.
# 

# C++
def codegen_Cxx_BASEDECL ( bases ) :
    if bases :
        bases = [ b.split('/')[-1] for b in bases ]
        return ': ' + ', '.join( [ 'public '+b for b in bases ] )

def codegen_Cxx_BASEINIT ( bases ) :
    if bases :
        bases = [ b.split('/')[-1] for b in bases ]
        return '  : ' + '\n  , '.join( [ b+'()' for b in bases ] )

def codegen_Cxx_BASEINCL ( bases ) :
    if bases :
        return '\n'.join( [ '#include "'+b+'.h"' for b in bases ] )

codegen_Cxx_template_BASEDECL = codegen_Cxx_BASEDECL
codegen_Cxx_template_BASEINIT = codegen_Cxx_BASEINIT
codegen_Cxx_template_BASEINCL = codegen_Cxx_BASEINCL
        
# python 
def codegen_python_BASEDECL ( bases ) :
    bases = [ b.split('.')[-1] for b in bases ]
    return '( ' + ', '.join(bases or ['object']) + ' )'

def codegen_python_BASEINIT ( bases ) :
    bases = [ b.split('.')[-1] for b in bases ]
    return '\n        '.join( [ b+'.__init__( self )' for b in bases ] )

def codegen_python_BASEINCL ( bases ) :
    res = []
    for b in bases :
        w = b.split('.')
        if len(w) == 1 :
            imp = 'from %s import %s' % ( b, b )
        else :
            imp = 'from %s import %s' % ( '.'.join(w[:-1]), w[-1] )
        res.append ( imp )
    return '\n'.join( res )

codegen_python_main_BASEDECL = codegen_python_BASEDECL
codegen_python_main_BASEINIT = codegen_python_BASEINIT
codegen_python_main_BASEINCL = codegen_python_BASEINCL

# descriptions of the templates
codegen_Cxx_description = "Template for regular non-templated C++ classes"
codegen_Cxx_template_description = "Template for templated C++ classes"
codegen_Cxx_app_description = "Template for C++ applications based on AppBase class"
codegen_Cxx_unit_test_description = "Template for C++ unit test module"

codegen_python_description = "Template for Python library module"
codegen_python_main_description = "Template for Python executable script"

codegen_README_description = "Template for README file"
codegen_ChangeLog_description = "Template for ChangeLog file"
codegen_SConscript_description = "Template for SConscript file for regular package"
