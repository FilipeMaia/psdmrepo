#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ConfigSvc...
#
#------------------------------------------------------------------------

"""Python wrapper for ConfigSvc class.

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

cdef extern from "<memory>" namespace "std":

    cdef cppclass auto_ptr[T]:
    
        auto_ptr()
        auto_ptr(T*)
        
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
    cdef void init(auto_ptr[ConfigSvcImplI]) except +


#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------

def initConfigSvc(object file):
    
    """initConfigSvc(file)\n\nInitialize configuration service, accepts file name or file-like object"""
    
    cdef auto_ptr[ConfigSvcImplI] ptr
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
            sstr = str(file)
            strfname = string(<char*>sstr)
        sstr = file.read()
        istr.str(string(<char*>sstr))
        ptr.reset(new ConfigSvcImplFile(istr, strfname))

    init(ptr)

cdef class ConfigSvc:

    cdef ConfigSvcPyHelper* thisptr      # hold a C++ instance which we're wrapping
    
    def __cinit__(self):
        self.thisptr = new ConfigSvcPyHelper();

    def __dealloc__(self):
        del self.thisptr

    
    def getStr(self, char* section, char* param, defval = None):        
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
        cdef string strsec = string(section)
        cdef string strparam = string(param)

        if defval is None:
            return self.thisptr.getBool(strsec, strparam)
        else:
            return PyBool_FromLong(self.thisptr.getBool(strsec, strparam, PyInt_AsLong(defval)))

    
    def getInt(self, char* section, char* param, defval = None):
        cdef string strsec = string(section)
        cdef string strparam = string(param)
        
        if defval is None:
            return self.thisptr.getInt(strsec, strparam)
        else:
            return self.thisptr.getInt(strsec, strparam, PyInt_AsLong(defval))
        
    
    def getFloat(self, char* section, char* param, defval = None):
        cdef string strsec = string(section)
        cdef string strparam = string(param)
        
        if defval is None:
            return self.thisptr.getDouble(strsec, strparam)
        else:
            return self.thisptr.getDouble(strsec, strparam, PyFloat_AsDouble(defval))


    def getStrList(self, char* section, char* param, list defval = None):
        cdef string strsec = string(section)
        cdef string strparam = string(param)
        cdef vector[string] tmpvec
        cdef object s
        cdef int i

        if defval is None:
            tmpvec = self.thisptr.getStrList(strsec, strparam)
        else:
            for s in defval:
                tmpvec.push_back(string(<char*>s))
            tmpvec = self.thisptr.getStrList(strsec, strparam, tmpvec)

        return [tmpvec[i].c_str() for i in range(tmpvec.size())]
        

    def getBoolList(self, char* section, char* param, list defval = None):
        cdef string strsec = string(section)
        cdef string strparam = string(param)
        cdef vector[int] tmpvec
        cdef int i

        if defval is None:
            tmpvec = self.thisptr.getBoolList(strsec, strparam)
        else:
            for i in defval:
                tmpvec.push_back(i)
            tmpvec = self.thisptr.getBoolList(strsec, strparam, tmpvec)

        return [PyBool_FromLong(tmpvec[i]) for i in range(tmpvec.size())]
        

    def getIntList(self, char* section, char* param, list defval = None):
        cdef string strsec = string(section)
        cdef string strparam = string(param)
        cdef vector[long] tmpvec
        cdef int i

        if defval is None:
            tmpvec = self.thisptr.getIntList(strsec, strparam)
        else:
            for i in defval:
                tmpvec.push_back(i)
            tmpvec = self.thisptr.getIntList(strsec, strparam, tmpvec)

        return [tmpvec[i] for i in range(tmpvec.size())]
        

    def getFloatList(self, char* section, char* param, list defval = None):
        cdef string strsec = string(section)
        cdef string strparam = string(param)
        cdef vector[double] tmpvec
        cdef float f
        cdef int i

        if defval is None:
            tmpvec = self.thisptr.getDoubleList(strsec, strparam)
        else:
            for f in defval:
                tmpvec.push_back(f)
            tmpvec = self.thisptr.getDoubleList(strsec, strparam, tmpvec)

        return [tmpvec[i] for i in range(tmpvec.size())]


    def put(self, char* section, char* param, object value):
        cdef string strsec = string(section)
        cdef string strparam = string(param)
        cdef object obj = PyObject_Str(value)
        cdef string str = string(<char*>obj)

        self.thisptr.put(strsec, strparam, str)

