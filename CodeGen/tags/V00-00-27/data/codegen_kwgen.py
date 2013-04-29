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
codegen_psana_module_description = "Template for PSANA user analysis module"

codegen_python_description = "Template for Python library module"
codegen_python_main_description = "Template for Python executable script"
codegen_python_unit_test_description = "Template for Python unit test script"
codegen_pyana_module_description = "Template for Pyana user analysis module"

codegen_README_description = "Template for README file"
codegen_ChangeLog_description = "Template for ChangeLog file"
codegen_SConscript_description = "Template for SConscript file for regular package"
codegen_SConscript_external_description = "Template for SConscript file for proxy package"

# define destinatio directory inside the package
# this is a dictonary indexed by language and extension
# if particular language or extension is not defined it is assumed
# that resulting file will go to the top directory
codegen_dstdir = {}

codegen_dstdir['Cxx'] = dict(cpp='src', h='include')
codegen_dstdir['Cxx_template'] = dict(cpp='src', h='include')
codegen_dstdir['psana_module'] = dict(cpp='src', h='include')
codegen_dstdir['Cxx_app'] = dict(cpp='app')
codegen_dstdir['Cxx_unit_test'] = dict(cpp='test')

codegen_dstdir['python'] = dict(py='src')
codegen_dstdir['python_main'] = {'': 'app'}
codegen_dstdir['python_unit_test'] = {'': 'test'}
codegen_dstdir['pyana_module'] = dict(py='src')

codegen_dstdir['ChangeLog'] = {'': 'doc'}
codegen_dstdir['README'] = {'': 'doc'}
