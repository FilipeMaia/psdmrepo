#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ConfigSvc...
#
#------------------------------------------------------------------------

"""Python wrapper for C++ ConfigSvc library.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id: template!python!py 4 2008-10-08 19:27:36Z salnikov $

@author Andy Salnikov
"""


#--------------------------------
#  Imports of standard modules --
#--------------------------------
from cpython.bool cimport *
from cpython.float cimport *
from cpython.int cimport *
from cpython.list cimport *
from cpython.object cimport *
from cpython.ref cimport *
from cpython.string cimport *
from libcpp.string cimport string
from libcpp.vector cimport vector

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

cdef extern from "<iostream>" namespace "std":

    cdef cppclass istream:
        pass

cdef extern from "<sstream>" namespace "std":

    cdef cppclass istringstream(istream):
    
        istringstream()
        istringstream(string)
        
        void str(string)
        string str()

cdef extern from "<boost/shared_ptr.hpp>" namespace "boost":

    cdef cppclass shared_ptr[T]:
    
        shared_ptr()
        shared_ptr(T*)
        
        T* get()
        void reset(T*)


cdef extern from "ConfigSvc/ConfigSvcImplI.h" namespace "ConfigSvc":

    cdef cppclass ConfigSvcImplI:
        pass
    
cdef extern from "ConfigSvc/ConfigSvcImplFile.h" namespace "ConfigSvc":

    cdef cppclass ConfigSvcImplFile(ConfigSvcImplI):
        
        ConfigSvcImplFile(string) except +
        ConfigSvcImplFile(istream, string) except +

cdef extern from "pyext/ConfigSvcPyHelper.h" namespace "ConfigSvc":

    cdef cppclass ConfigSvcPyHelper:
            
        ConfigSvcPyHelper()
        
        int getBool(string, string) except +
        long getInt(string, string) except +
        double getDouble(string, string) except +
        string getStr(string, string) except +

        int getBool(string, string, int) except +
        long getInt(string, string, long) except +
        double getDouble(string, string, double) except +
        string getStr(string, string, string) except +

        vector[int] getBoolList(string, string) except +
        vector[long] getIntList(string, string) except +
        vector[double] getDoubleList(string, string) except +
        vector[string] getStrList(string, string) except +

        vector[int] getBoolList(string, string, vector[int]) except +
        vector[long] getIntList(string, string, vector[long]) except +
        vector[double] getDoubleList(string, string, vector[double]) except +
        vector[string] getStrList(string, string, vector[string]) except +
        
        void put(string, string, string) except +

cdef extern from "ConfigSvc/ConfigSvc.h" namespace "ConfigSvc::ConfigSvc":

    # this is actually a static method in ConfigSvc::ConfigSvc class
    cdef void init(shared_ptr[ConfigSvcImplI]) except +


#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------

def initConfigSvc(object file not None):
    """
    initConfigSvc(file)
    
    Initialize configuration service, accepts file name or file-like object.
    If *file* parameter has string type then it is assumed to be a file name,
    otherwise it is expected to be a file of file-like object which defines 
    ``read()`` method.
    
    File contents is read and its contents is used for configuration parameters.
    In case of any errors ``RuntimeError`` exception is raised.
    """
    
    cdef shared_ptr[ConfigSvcImplI] ptr
    cdef string strfname
    cdef object sstr
    cdef istringstream istr

    if PyString_Check(file):
        
        # if string is passed then it is a file name
        strfname = string(PyString_AsString(file))
        ptr.reset(new ConfigSvcImplFile(strfname))
        
    else:
        
        # otherwise it must be a file-like object, read it into memory
        if hasattr(file, 'name'):
            strfname = string(<char*>file.name)
        else:
            sstr = PyObject_Str(file)
            strfname = string(<char*>sstr)
        sstr = file.read()
        istr.str(string(<char*>sstr))
        ptr.reset(new ConfigSvcImplFile(istr, strfname))

    init(ptr)

cdef class ConfigSvc:
    """
    Python wrapper for C++ class ConfigSvc. It does not provide exact same 
    signatures for methods because C++ methods are all template-based, instead
    this class defines more "Pythonic" interfaces to the same underlying C++
    service.
    
    Note that before you can use any of the methods in this class configuration
    service must be initialized. This is done with :py:func:`initConfigSvc` function 
    defined in this module. If service is not initialized then calling any method
    will result in ``RuntimeError`` exception.
    """

    cdef ConfigSvcPyHelper* thisptr      # hold a C++ instance which we're wrapping
    
    def __cinit__(self):
        self.thisptr = new ConfigSvcPyHelper();

    def __dealloc__(self):
        del self.thisptr

    
    def getStr(self, char* section, char* param, defval = None):
        """
        self.getStr(section: str, param: str, defval: str = None) -> str
        
        Returns string value of the requested parameter. Takes section name and 
        parameter name which are both strings. If *defval* parameter is ``None``
        or missing then parameter must be defined; if parameter was not defined
        then ``RuntimeError`` is raised. If *defval* is not ``None`` then 
        it must have string type and it will be returned when parameter is not
        defined.
        """
        cdef string strsec = string(section)
        cdef string strparam = string(param)
        cdef string tmpstr

        if defval is None:
            tmpstr = self.thisptr.getStr(strsec, strparam)
        else:
            tmpstr = string(PyString_AsString(defval))
            tmpstr = self.thisptr.getStr(strsec, strparam, tmpstr)
        return tmpstr.c_str()

    
    def getBool(self, char* section, char* param, defval = None):
        """
        self.getBool(section: str, param: str, defval: int = None) -> bool
        
        Returns boolean value of the requested parameter. Takes section name and 
        parameter name which are both strings. If *defval* parameter is ``None``
        or missing then parameter must be defined; if parameter was not defined
        then ``RuntimeError`` is raised. If *defval* is not ``None`` then 
        it must have int or bool type and its boolean value will be returned when 
        parameter is not defined.
        """
        cdef string strsec = string(section)
        cdef string strparam = string(param)

        if defval is None:
            return self.thisptr.getBool(strsec, strparam)
        else:
            return PyBool_FromLong(self.thisptr.getBool(strsec, strparam, PyInt_AsLong(defval)))

    
    def getInt(self, char* section, char* param, defval = None):
        """
        self.getInt(section: str, param: str, defval: int = None) -> int
        
        Returns integer value of the requested parameter. Takes section name and 
        parameter name which are both strings. If *defval* parameter is ``None``
        or missing then parameter must be defined; if parameter was not defined
        then ``RuntimeError`` is raised. If *defval* is not ``None`` then 
        it must have int type and its value will be returned when parameter is 
        not defined.
        """
        cdef string strsec = string(section)
        cdef string strparam = string(param)
        
        if defval is None:
            return self.thisptr.getInt(strsec, strparam)
        else:
            return self.thisptr.getInt(strsec, strparam, PyInt_AsLong(defval))
        
    
    def getFloat(self, char* section, char* param, defval = None):
        """
        self.getFloat(section: str, param: str, defval: float = None) -> float
        
        Returns floating point value of the requested parameter. Takes section 
        name and parameter name which are both strings. If *defval* parameter is 
        ``None`` or missing then parameter must be defined; if parameter was not 
        defined then ``RuntimeError`` is raised. If *defval* is not ``None`` 
        then it must have float type and its value will be returned when parameter 
        is not defined.
        """
        cdef string strsec = string(section)
        cdef string strparam = string(param)
        
        if defval is None:
            return self.thisptr.getDouble(strsec, strparam)
        else:
            return self.thisptr.getDouble(strsec, strparam, PyFloat_AsDouble(defval))


    def getStrList(self, char* section, char* param, list defval = None):
        """
        self.getStrList(section: str, param: str, defval: list = None) -> list of strings
        
        Returns value of the requested parameter as list of strings. Takes section 
        name and parameter name which are both strings. If *defval* parameter is 
        ``None`` or missing then parameter must be defined; if parameter was not 
        defined then ``RuntimeError`` is raised. If *defval* is not ``None`` 
        then it must have list type and all its items must have string type. 
        Its value will be returned when parameter is not defined.
        """
        cdef string strsec = string(section)
        cdef string strparam = string(param)
        cdef vector[string] tmpvec
        cdef object s
        cdef unsigned i

        if defval is None:
            tmpvec = self.thisptr.getStrList(strsec, strparam)
        else:
            for s in defval:
                tmpvec.push_back(string(<char*>s))
            tmpvec = self.thisptr.getStrList(strsec, strparam, tmpvec)

        return [tmpvec[i].c_str() for i in range(tmpvec.size())]
        

    def getBoolList(self, char* section, char* param, list defval = None):
        """
        self.getBoolList(section: str, param: str, defval: list = None) -> list of bool
        
        Returns value of the requested parameter as list of booleans. Takes section 
        name and parameter name which are both strings. If *defval* parameter is 
        ``None`` or missing then parameter must be defined; if parameter was not 
        defined then ``RuntimeError`` is raised. If *defval* is not ``None`` 
        then it must have list type and all its items must have int or bool type. 
        Its value will be returned when parameter is not defined.
        """
        cdef string strsec = string(section)
        cdef string strparam = string(param)
        cdef vector[int] tmpvec
        cdef int i
        cdef unsigned j

        if defval is None:
            tmpvec = self.thisptr.getBoolList(strsec, strparam)
        else:
            for i in defval:
                tmpvec.push_back(i)
            tmpvec = self.thisptr.getBoolList(strsec, strparam, tmpvec)

        return [PyBool_FromLong(tmpvec[j]) for j in range(tmpvec.size())]
        

    def getIntList(self, char* section, char* param, list defval = None):
        """
        self.getIntList(section: str, param: str, defval: list = None) -> list of ints
        
        Returns value of the requested parameter as list of ints. Takes section 
        name and parameter name which are both strings. If *defval* parameter is 
        ``None`` or missing then parameter must be defined; if parameter was not 
        defined then ``RuntimeError`` is raised. If *defval* is not ``None`` 
        then it must have list type and all its items must have int type. 
        Its value will be returned when parameter is not defined.
        """
        cdef string strsec = string(section)
        cdef string strparam = string(param)
        cdef vector[long] tmpvec
        cdef int i
        cdef unsigned j

        if defval is None:
            tmpvec = self.thisptr.getIntList(strsec, strparam)
        else:
            for i in defval:
                tmpvec.push_back(i)
            tmpvec = self.thisptr.getIntList(strsec, strparam, tmpvec)

        return [tmpvec[j] for j in range(tmpvec.size())]
        

    def getFloatList(self, char* section, char* param, list defval = None):
        """
        self.getFloatList(section: str, param: str, defval: list = None) -> list of floats
        
        Returns value of the requested parameter as list of floats. Takes section 
        name and parameter name which are both strings. If *defval* parameter is 
        ``None`` or missing then parameter must be defined; if parameter was not 
        defined then ``RuntimeError`` is raised. If *defval* is not ``None`` 
        then it must have list type and all its items must have float type. 
        Its value will be returned when parameter is not defined.
        """
        cdef string strsec = string(section)
        cdef string strparam = string(param)
        cdef vector[double] tmpvec
        cdef float f
        cdef unsigned i

        if defval is None:
            tmpvec = self.thisptr.getDoubleList(strsec, strparam)
        else:
            for f in defval:
                tmpvec.push_back(f)
            tmpvec = self.thisptr.getDoubleList(strsec, strparam, tmpvec)

        return [tmpvec[i] for i in range(tmpvec.size())]


    def put(self, char* section, char* param, object value):
        """
        self.put(section: str, param: str, value: object)

        Defines or updates parameter value.  Takes section name and parameter 
        name which are both strings. *value* argument is converted to string
        (with usual ``str(value)`` function) and the resulting value is stored 
        as new parameter value.
        """
        cdef string strsec = string(section)
        cdef string strparam = string(param)
        cdef object obj = PyObject_Str(value)
        cdef string str = string(<char*>obj)

        self.thisptr.put(strsec, strparam, str)

