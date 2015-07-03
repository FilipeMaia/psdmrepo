#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ConfigParameters...
#
#------------------------------------------------------------------------

"""Is a base class with infrastructure for storage for configuration parameters.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import os
import sys

#-----------------------------
# Imports for other modules --
#-----------------------------
from Logger import logger

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#---------------------
#  Class definition --
#---------------------

class Parameter :
    """Single parameters.
    #@see OtherClass ConfigParameters
    #@see OtherClass ConfigParametersCorAna
    """

    dicBool = {'false':False, 'true':True}

    _name      = 'EMPTY'
    _type      = None
    _value     = None
    _value_def = None
    _index     = None

#---------------------

    def __init__ ( self, name='EMPTY', val=None, val_def=None, type='str', index=None) :
        """Constructor.
        @param name    parameter name
        @param val     parameter value
        @param val_def parameter default value
        @param type    parameter type, implemented types: 'str', 'int', 'long', 'float', 'bool'
        @param index   parameter index the list
        """
        self.setParameter ( name, val, val_def, type, index ) 

#---------------------

    def setParameter ( self, name='EMPTY', val=None, val_def=None, type='str', index=None ) :
        self._value_def = val_def
        self._name      = name
        self._type      = type
        self._index     = index
        self.setValue ( val )

#---------------------

    def setValue ( self, val=None ) :
        if val is None :
            self._value = self._value_def
        else :
            if   self._type == 'str' :
                self._value = str( val )
        
            elif self._type == 'int' :
                self._value = int( val )
        
            elif self._type == 'long' :
                self._value = long( val )
        
            elif self._type == 'float' :
                self._value = float( val )
        
            elif self._type == 'bool' :
                self._value = bool( val )
            else : 
                self._value = val

#---------------------

    def setDefaultValue ( self ) :
        self._value = self._value_def

#---------------------

    def setDefault (self) :
        self._value = self._value_def

#---------------------

    def setValueFromString ( self, str_val ) :
        """Set parameter value fron string based on its declared type: 'str', 'int', 'long', 'float', 'bool' """

        if str_val.lower() == 'none' :
            self._value = self._value_def

        if self._type == 'str' :
            self._value = str( str_val )

        elif self._type == 'int' :
            self._value = int( str_val )

        elif self._type == 'long' :
            self._value = long( str_val )

        elif self._type == 'float' :
            self._value = float( str_val )

        elif self._type == 'bool' :
            self._value = self.dicBool[str_val.lower()]

        else :
            msg = 'Parameter.setValueForType: Requested parameter type ' + type + ' is not supported\n'  
            msg+= 'WARNING! The parameter value is left unchanged...\n'
            logger.warning(msg)

#---------------------

    def setType ( self, type='str' ) :
        self._type = type

    def setName ( self, name='EMPTY' ) :
        self._name = name

    def value ( self ) :
        return self._value

    def value_def ( self ) :
        return self._value_def

    def name ( self ) :
        return self._name

    def type ( self ) :
        return self._type

    def index( self ) :
        return self._index

#---------------------

    def strParInfo( self ) :
        s = 'Par: %s %s %s %s' % ( self.name().ljust(32), str(self.value()).ljust(32), self.type().ljust(8), str(self.index()).ljust(8) )
        return s

#---------------------

    def printParameter( self ) :
        s = self.strParInfo()
        logger.info( s )
        print s

#---------------------
#---------------------
#---------------------
#---------------------

class ConfigParameters :
    """Is intended as a storage for configuration parameters.
    #@see OtherClass ConfigParametersCorana
    """

    dict_pars  = {} # Dictionary for all configuration parameters, containing pairs {<parameter-name>:<parameter-object>, ... } 
    dict_lists = {} # Dictionary for declared lists of configuration parameters:    {<list-name>:<list-of-parameters>, ...}

    def __init__ ( self, fname=None ) :
        """Constructor.
        @param fname  the file name with configuration parameters, if not specified then it will be set to the default value at declaration.
        """

        self.fname_cp = self.declareParameter( name='FNAME_CONFIG_PARS', val=fname, val_def='confpars.txt', type='str' ) 

#---------------------------------------

    def declareParameter( self, name='EMPTY', val=None, val_def=None, type='str', index=None ) :
        par = Parameter( name, val, val_def, type, index )
        #self.dict_pars[name] = par
        self.dict_pars.update( {name:par} )
        return par

#---------------------------------------

    def declareListOfPars( self, list_name='EMPTY_LIST', list_val_def_type=None ) :
        list_of_pars = []

        if list_val_def_type is None : return None

        for index,rec in enumerate(list_val_def_type) :
            name = list_name + ':' + str(index)
            val, val_def, type = rec

            #par = self.declareParameter( name, val, val_def, type, index )
            par = Parameter( name, val, val_def, type, index )
            list_of_pars.append(par)
            self.dict_pars.update( {name:par} )

        self.dict_lists.update( {list_name:list_of_pars} )

        return list_of_pars

#---------------------------------------

    def getListOfPars( self, name ) :
        return self.dict_lists[name]

#---------------------------------------

    def printListOfPars( self, name ) :
        list_of_pars = self.getListOfPars(name)

        print 'Parameters for list:', name
        for par in list_of_pars :
            par.printParameter()

#---------------------------------------

    def printParameters( self ) :
        msg = 'printParameters - Number of declared parameters in the dict: %d' % len(self.dict_pars)
        logger.info(msg, __name__)
        for par in self.dict_pars.values() :
            s = par.strParInfo()
            logger.info( s )
            #print s

#---------------------------------------

    def setDefaultValues( self ) :
        for par in self.dict_pars.values() :
            par.setDefaultValue()

#---------------------------------------

    def setParsFileName(self, fname=None) :
        if fname is None :
            self.fname = self.fname_cp.value()
        else :
            self.fname = fname

#---------------------------------------

    def saveParametersInFile ( self, fname=None ) :
        self.setParsFileName(fname)        
        logger.info('Save configuration parameters in file ' + self.fname, __name__)
        f=open(self.fname,'w')
        for par in self.dict_pars.values() :

            v = par.value()
            #if par.type() == 'int' : v = int( v )

            s = '%s %s\n' % ( par.name().ljust(32), str(v) )
            f.write( s )
        f.close() 

#---------------------------------------

    def setParameterValueByName ( self, name, str_val ) :

        if not ( name in self.dict_pars.keys() ) :
            msg  = 'The parameter name ' + name + ' is unknown in the dictionary.\n'
            msg += 'WARNING! Parameter needs to be declared first. Skip this parameter initialization.\n' 
            logger.warning(msg)
            return

        self.dict_pars[name].setValueFromString(str_val)

#---------------------------------------

    def readParametersFromFile ( self, fname=None ) :
        self.setParsFileName(fname)        
        logger.info('Read config. pars from file ' + self.fname, __name__)

        if not os.path.exists(self.fname) :
            logger.warning('readParametersFromFile : The file ' + self.fname + ' is not found', __name__)
            return
 
        f=open(self.fname,'r')
        for line in f :
            if len(line) == 1 : continue # line is empty
            fields = line.rstrip('\n').split(' ',1)
            self.setParameterValueByName ( name=fields[0], str_val=fields[1].strip(' ') )
        f.close() 

#---------------------------------------

def usage() :
    msg  = 'Use command: ' + sys.argv[0] + ' [<configuration-file-name>]\n'
    msg += 'with a single or without arguments.' 
    msg = '\n' + 51*'-' + '\n' + msg + '\n' + 51*'-'
    logger.warning(msg, __name__)

#---------------------------------------

def getConfigFileFromInput() :
    """DO NOT PARSE INPUT PARAMETERS IN THIS APPLICATION
    This is interfere with other applications which really need to use input pars,
    for example maskeditor...
    """

    return None


    msg = 'Input pars sys.argv: '
    for par in sys.argv :  msg += par
    logger.debug(msg, __name__)

    if len(sys.argv) > 2 : 
        usage()
        msg  = 'Too many arguments ...\n'
        msg += 'EXIT application ...\n'
        sys.exit (msg)

    elif len(sys.argv) == 1 : 
        return None

    else :
        path = sys.argv[1]        
        if os.path.exists(path) :
            return path
        else :
            usage()
            msg  = 'Requested configuration file "' + path + '" does not exist.\n'
            msg += 'EXIT application ...\n'
            sys.exit (msg)


#---------------------------------------

# confpars = ConfigParameters ( getConfigFileFromInput() ) # is moved to subclass like ConfigParametersCorAna

#---------------------------------------

if __name__ == "__main__" :
    sys.exit ( "Module is not supposed to be run as main module" )

#---------------------------------------
