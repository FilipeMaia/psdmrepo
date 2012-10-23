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

#---------------------

    def __init__ ( self, name='EMPTY', val=None, val_def=None, typ='str' ) :
        """Constructor.
        @param name    parameter name
        @param val     parameter value
        @param val_def parameter default value
        @param typ     parameter type, implemented types: 'str', 'int', 'long', 'float', 'bool'
        """
        self.setParameter ( name, val, val_def, typ ) 

#---------------------

    def setParameter ( self, name='EMPTY', val=None, val_def=None, typ='str' ) :
        self._value_def = val_def
        self._name      = name
        self._type      = typ
        self.setValue ( val )

#---------------------

    def setValue ( self, val=None ) :
        if val == None :
            self._value = self._value_def
        else :
            self._value = val

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
            self._value = dicBool[str_val.lower()]

        else :
            msg = 'Parameter.setValueForType: Requested parameter type ' + typ + ' is not supported\n'  
            msg+= 'WARNING! The parameter value is left unchanged...\n'
            logger.warning(msg)

#---------------------

    def setType ( self, typ='str' ) :
        self._type = typ

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

#---------------------
#---------------------
#---------------------
#---------------------

class ConfigParameters :
    """Is intended as a storage for configuration parameters.
    #@see OtherClass ConfigParametersCorana
    """

    dict_pars = {} # Dictionary for all configuration parameters, consisting of pairs {<parameter-name>:<parameter-object>, ... } 

    def __init__ ( self, fname=None ) :
        """Constructor.
        @param fname  the file name with configuration parameters, if not specified then it will be set to the default value at declaration.
        """

        self.fname_cp = self.declareParameter( name='FNAME_CONFIG_PARS', val=fname, val_def='confpars.txt', typ='str' ) 

#---------------------------------------

    def declareParameter( self, name='EMPTY', val=None, val_def=None, typ='str' ) :
        par = Parameter( name, val, val_def, typ )
        self.dict_pars[name] = par
        return par

#---------------------------------------

    def setDefaultValues( self ) :
        for par in self.dict_pars.values() :
            par.setValue ( val=None )

#---------------------------------------

    def printParameters( self ) :
        print 'Number of declared parameters in the dict:', len(self.dict_pars)
        for par in self.dict_pars.values() :
            s = 'Par: %s %s %s' % ( par.name().ljust(32), str(par.value()).ljust(32), par.type() )
            print s

#---------------------------------------

    def setParsFileName(self, fname=None) :
        if fname == None :
            self.fname = self.fname_cp.value()
        else :
            self.fname = fname

#---------------------------------------

    def saveParametersInFile ( self, fname=None ) :
        self.setParsFileName(fname)        
        logger.info('Save configuration parameters in file ' + self.fname)
        f=open(self.fname,'w')
        for par in self.dict_pars.values() :
            s = '%s %s\n' % ( par.name().ljust(32), str(par.value()) )
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
        logger.info('Read configuration parameters from file ' + self.fname)

        if not os.path.exists(self.fname) :
            logger.warning('readParametersFromFile : The file ' + self.fname + ' is not found')
            return
 
        f=open(self.fname,'r')
        for line in f :
            if len(line) == 1 : continue # line is empty
            fields = line.split()
            self.setParameterValueByName ( name=fields[0], str_val=fields[1] )
        f.close() 

#---------------------------------------

def usage() :
    msg  = 'Use command: ' + sys.argv[0] + ' [<configuration-file-name>]\n'
    msg += 'with a single or without arguments.' 
    msg = '\n' + 51*'-' + '\n' + msg + '\n' + 51*'-'
    logger.warning(msg)

#---------------------------------------

def getConfigFileFromInput() :

    msg = 'List of input parameters: '
    for par in sys.argv :  msg += par
    logger.info(msg)

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
