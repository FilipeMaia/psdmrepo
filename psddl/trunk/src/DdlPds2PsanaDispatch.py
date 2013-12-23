#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module DdlPds2Psana...
#
#------------------------------------------------------------------------

"""DDL parser which generates pds2psana C++ code.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Andrei Salnikov
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
import types
import collections

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
import jinja2 as ji
from psddl.Package import Package
from psddl.Type import Type
from psddl.Template import Template as T
from psddl.TemplateLoader import TemplateLoader

#----------------------------------
# Local non-exported definitions --
#----------------------------------

# list of type IDs which are not needed or handled separately,
# we add these to the switch to suppress warnings. If one of these
# types gets implemented it should be removed from the list.
_ignored_types = [
        'Any',
        'Id_Xtc', 
        'NumberOf', 
        'Id_Epics', 
        'Reserved1',
        'Reserved2',
        'Id_Index',
        'Id_XampsConfig',
        'Id_XampsElement',
        'Id_FexampConfig',
        'Id_FexampElement',
        'Id_PhasicsConfig',
        'Id_CspadCompressedElement',
        ]

# Extra headers needed for special proxy classes of similar stuff
_extra_headers = [
        "psddl_pds2psana/CsPadDataOrdered.h",
        "psddl_pds2psana/PnccdFullFrameV1Proxy.h",
        "psddl_pds2psana/TimepixDataV1ToV2.h",
        ]

# types that need special UseSize argument for proxy, 
# there should not be too many of these 
_use_size_types = [
        "Acqiris::TdcDataV1"
        ]

# some types need to substitute generated final classes with 
# hand-written stuff
def _finalClass(type, final_ns, cfgType=None):
    typeName = type.fullName('C++')
    if typeName.startswith("CsPad::DataV"):
        # cspad need special final type
        version = typeName[12:]
        ns = final_ns + '::' if final_ns else ''
        typeName = "{0}CsPadDataOrdered<{0}CsPad::DataV{1}<{2}>, Psana::CsPad::ElementV{1}>".format(ns, version, cfgType)
    elif typeName == "Timepix::DataV1":
        ns = final_ns + '::' if final_ns else ''
        typeName = ns + "TimepixDataV1ToV2"
    else:
        # for all other types use generated stuff
        typeName = type.fullName('C++', final_ns)
        if cfgType: typeName = "{0}<{1}>".format(typeName, cfgType)
    return typeName

# some types convert XTC types into different (versions) of psana types
def _psanaClass(type, psana_ns):
    typeName = type.fullName('C++')
    if typeName == "Timepix::DataV1":
        ns = psana_ns + '::' if psana_ns else ''
        typeName = ns + "Timepix::DataV2"
    else:
        # for all other types use generated stuff
        typeName = type.fullName('C++', psana_ns)
    return typeName

# some types need special proxy class
def _proxyClass(type, psana_type, final_type, xtc_type, config_type=None):

    typeName = type.fullName('C++')
    if typeName == 'PNCCD::FullFrameV1':
        return 'PnccdFullFrameV1Proxy'
    elif not type.xtcConfig:
        use_size = type.fullName('C++') in _use_size_types
        if use_size:
            return 'EvtProxy<{0}, {1}, {2}, true>'.format(psana_type, final_type, xtc_type)
        else:
            return 'EvtProxy<{0}, {1}, {2}>'.format(psana_type, final_type, xtc_type)
    else:
            return 'EvtProxyCfg<{0}, {1}, {2}, {3}>'.format(psana_type, final_type, xtc_type, config_type)

# jinja environment
_jenv = ji.Environment(loader=TemplateLoader(), trim_blocks=True,
                       line_statement_prefix='$', line_comment_prefix='$$')

def _TEMPL(template):
    return _jenv.get_template('pds2psana_dispatch.tmpl?'+template)

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class DdlPds2PsanaDispatch ( object ) :

    @staticmethod
    def backendOptions():
        """ Returns the list of options supported by this backend, returned value is 
        either None or a list of triplets (name, type, description)"""
        return [
            ('psana-ns', 'STRING', "namespace for Psana types, default: Psana"),
            ('pdsdata-ns', 'STRING', "namespace for pdsdata types, default: Pds"),
            ]

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, backend_options, log ) :
        '''Constructor
        
           @param backend_options  dictionary of options passed to backend
           @param log              message logger instance
        '''
        self.incname = backend_options['global:header']
        self.cppname = backend_options['global:source']
        self.incdirname = backend_options.get('global:gen-incdir', "")
        self.top_pkg = backend_options.get('global:top-package')
        
        self.psana_ns = backend_options.get('psana-ns', "Psana")
        self.pdsdata_ns = backend_options.get('pdsdata-ns', "Pds")

        self._types = {}

        self._log = log

        #include guard
        g = os.path.split(self.incname)[1]
        if self.top_pkg: g = self.top_pkg + '_' + g
        self.guard = g.replace('.', '_').upper()

    #-------------------
    #  Public methods --
    #-------------------

    def parseTree ( self, model ) :
        
        # open output files
        self.inc = file(self.incname, 'w')
        self.cpp = file(self.cppname, 'w')

        # loop over packages and types in the model
        for ns in model.namespaces() :
            if isinstance(ns, Package) :
                if not ns.included :
                    self._parsePackage(ns)
            elif isinstance(ns, Type) :
                if not ns.external:
                    self._parseType(type = ns)

        # generate code for all collected types
        types, headers, psana_types = self._codegen()

        # add own header to the list
        headers = [os.path.join(self.incdirname, os.path.basename(self.incname))] + list(headers) + _extra_headers

        # write the dispatch function
        inc_guard = self.guard
        namespace = self.top_pkg
        ignored_types = _ignored_types

        # generate code for typeInfoPtrs function
        typeInfoPtrsCode = _TEMPL('typeinfoptrs').render(locals())

        print >>self.inc, _TEMPL('header_template').render(locals())
        print >>self.cpp, _TEMPL('impl_template').render(locals())
        
        # close dispatch function files
        self.inc.close()
        self.cpp.close()


    def _parsePackage(self, pkg):

        # loop over packages and types
        for ns in pkg.namespaces() :
            
            if isinstance(ns, Package) :
                
                self._parsePackage(ns)
            
            elif isinstance(ns, Type) :
    
                self._parseType(type = ns)


    def _parseType(self, type):

        self._log.debug("_parseType: type=%s", repr(type))

        if type.type_id is None: return
        
        self._types.setdefault(type.type_id, []).append(type)


    def _codegen(self):
        ''' 
        Retuns tuple containing three elements:
        1. Dictinary mappig TypeId type name (like 'Id_AcqConfig') to the corresponding piece of code
        2. List of heder names to be included 
        3. 2d dictionary mapping TypeId/version to a list of Psana types that the dispatch code 
           returned in 1. will put in the event and config store
        ''' 
        codes = {}
        headers = set()

        # make a 2D dictionary with appropriate default values
        def callableDefaultDictList():
            return collections.defaultdict(list)
        psana_types = collections.defaultdict(callableDefaultDictList)

        for type_id, types in self._types.items():

            versions = {}
            for type in types:
                
                code, header, psana_type = self._typecode(type)
                headers.add(header)
                
                v = int(type.version)
                versions.setdefault(v, []).append(code)
                psana_types[type_id][v].append(psana_type)
                # for event-type data add compressed version as well
                if 'config-type' not in type.tags:
                    v = int(type.version) | 0x8000
                    versions.setdefault(v, []).append(code)
                    psana_types[type_id][v].append(psana_type)
                
            codes[type_id] = _TEMPL('version_switch_template').render(locals())

        return codes, headers, psana_types


    def _typecode(self, type):
        '''
        For a given type returns tuple of two elements:
        1. Piece of code which produces final objects
        2. Name of the include file
        3. psana type put into event or config store
        '''
        header = os.path.basename(type.location)
        header = os.path.splitext(header)[0]
        if not header.endswith('.ddl'): header += '.ddl'
        header = header + '.h'
        header = os.path.join(self.incdirname, header)
        
        xtc_type = type.fullName('C++', self.pdsdata_ns)
        psana_type = _psanaClass(type, self.psana_ns)
 
        code = ""
        
        if 'config-type' in type.tags:
            # config types
            if type.value_type:
                final_namespace = type.parent.fullName('C++', self.top_pkg)                
                code = _TEMPL('config_value_store_template').render(locals())
            else:
                final_type = _finalClass(type, self.top_pkg)
                code = _TEMPL('config_abs_store_template').render(locals())
        else:
            # non-config types
            use_size = type.fullName('C++') in _use_size_types
            if type.value_type:
                final_namespace = type.parent.fullName('C++', self.top_pkg)                
                code = _TEMPL('event_value_store_template').render(locals())
            elif not type.xtcConfig:
                final_type = _finalClass(type, self.top_pkg)
                proxy_type = _proxyClass(type, psana_type, final_type, xtc_type)
                code = _TEMPL('event_abs_store_template').render(locals())
            else:
                config_types = {}
                for t in type.xtcConfig:
                    cfg_type = t.fullName('C++', self.pdsdata_ns)
                    final_type = _finalClass(type, self.top_pkg, cfg_type)
                    config_types[cfg_type] = _proxyClass(type, psana_type, final_type, xtc_type, cfg_type)
                code = _TEMPL('event_cfg_store_template').render(locals())

        return code, header, psana_type

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
