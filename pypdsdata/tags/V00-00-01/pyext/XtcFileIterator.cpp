//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcFileIterator...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcFileIterator.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Dgram.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  int XtcFileIterator_init( PyObject* self, PyObject* args, PyObject* kwds );
  void XtcFileIterator_dealloc( PyObject* self );

  // type-specific methods
  PyObject* XtcFileIterator_iter( PyObject* self );
  PyObject* XtcFileIterator_next( PyObject* self );

  PyMethodDef XtcFileIterator_Methods[] = {
    {0, 0, 0, 0}
   };

  char XtcFileIterator_doc[] = "Python class wrapping C++ Pds::XtcFileIterator class.\n\n"
      "Constructor of the class has one argument which is a Python file object. The\n"
      "instances of this class act as regular Python iterators which return objects\n"
      "of type Dgram.";

  PyTypeObject XtcFileIterator_Type = {
    PyObject_HEAD_INIT(0)
    0,                       /*ob_size*/
    "pdsdata.xtc.XtcFileIterator", /*tp_name*/
    sizeof(pypdsdata::XtcFileIterator), /*tp_basicsize*/
    0,                       /*tp_itemsize*/
    /* methods */
    XtcFileIterator_dealloc, /*tp_dealloc*/
    0,                       /*tp_print*/
    0,                       /*tp_getattr*/
    0,                       /*tp_setattr*/
    0,                       /*tp_compare*/
    0,                       /*tp_repr*/
    0,                       /*tp_as_number*/
    0,                       /*tp_as_sequence*/
    0,                       /*tp_as_mapping*/
    0,                       /*tp_hash*/
    0,                       /*tp_call*/
    0,                       /*tp_str*/
    PyObject_GenericGetAttr, /*tp_getattro*/
    PyObject_GenericSetAttr, /*tp_setattro*/
    0,                       /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,      /*tp_flags*/
    XtcFileIterator_doc,     /*tp_doc*/
    0,                       /*tp_traverse*/
    0,                       /*tp_clear*/
    0,                       /*tp_richcompare*/
    0,                       /*tp_weaklistoffset*/
    XtcFileIterator_iter,    /*tp_iter*/
    XtcFileIterator_next,    /*tp_iternext*/
    XtcFileIterator_Methods, /*tp_methods*/
    0,                       /*tp_members*/
    0,                       /*tp_getset*/
    0,                       /*tp_base*/
    0,                       /*tp_dict*/
    0,                       /*tp_descr_get*/
    0,                       /*tp_descr_set*/
    0,                       /*tp_dictoffset*/
    XtcFileIterator_init,    /*tp_init*/
    PyType_GenericAlloc,     /*tp_alloc*/
    PyType_GenericNew,       /*tp_new*/
    _PyObject_Del,           /*tp_free*/
    0,                       /*tp_is_gc*/
    0,                       /*tp_bases*/
    0,                       /*tp_mro*/
    0,                       /*tp_cache*/
    0,                       /*tp_subclasses*/
    0,                       /*tp_weaklist*/
    XtcFileIterator_dealloc  /*tp_del*/
  };


  // Destructor function for Dgrams
  void dgramDtor(Pds::Dgram* dgram) {
    delete[] (char*)(dgram);
  }
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

namespace pypdsdata {

PyTypeObject*
XtcFileIterator::typeObject()
{
  return &::XtcFileIterator_Type;
}


/// factory function
XtcFileIterator*
XtcFileIterator::XtcFileIterator_FromFile( PyObject* file )
{
  XtcFileIterator* ob = PyObject_New(XtcFileIterator,typeObject());
  if ( not ob ) {
    PyErr_SetString( PyExc_RuntimeError, "Failed to create XtcFileIterator object." );
    return 0;
  }

  Py_XINCREF(file);
  ob->m_file = file;
  ob->m_count = 0;

  return ob;
}

} // namespace pypdsdata


namespace {

int
XtcFileIterator_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::XtcFileIterator* py_this = (pypdsdata::XtcFileIterator*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  PyObject* fileObj ;
  if ( not PyArg_ParseTuple( args, "O:XtcFileIterator", &fileObj ) ) return -1;

  if ( not PyFile_Check( fileObj ) ) {
    PyErr_SetString(PyExc_TypeError, "Error: XtcFileIterator expects file object");
    return -1;
  }

  Py_CLEAR(py_this->m_file);
  Py_INCREF(fileObj);
  py_this->m_file = fileObj;
  py_this->m_count = 0;

  return 0;
}


void
XtcFileIterator_dealloc( PyObject* self )
{
  pypdsdata::XtcFileIterator* py_this = (pypdsdata::XtcFileIterator*) self;

  // free the file object from us
  Py_XDECREF(py_this->m_file);

  // deallocate ourself
  self->ob_type->tp_free(self);
}


PyObject* XtcFileIterator_iter( PyObject* self )
{
  Py_XINCREF(self);
  return self;
}

PyObject* XtcFileIterator_next( PyObject* self )
{
  pypdsdata::XtcFileIterator* py_this = (pypdsdata::XtcFileIterator*) self;

  FILE* file = PyFile_AsFile( py_this->m_file );

  // read header
  Pds::Dgram header;
  if ( fread(&header, sizeof(header), 1, file) != 1 ) {
    if ( feof(file) ) {
      // signal end of iteration
      PyErr_SetNone( PyExc_StopIteration );
      return 0;
    } else {
      // something bad happened
      PyObject* fnameObj = PyFile_Name( py_this->m_file );
      PyErr_SetFromErrnoWithFilename( PyExc_IOError, PyString_AsString(fnameObj) );
      return 0;
    }
  }

  // get the data size and allocate whole buffer
  size_t payloadSize = header.xtc.sizeofPayload();
  size_t dgramSize = sizeof(header) + payloadSize;
  char* buf ;
  try {
    buf = new char[dgramSize];
  } catch ( ... ) {
    PyErr_Format(PyExc_MemoryError, "Error: failed to allocate buffer memory");
    return 0;
  }

  // copy header into new buffer
  std::copy( (char*)&header, ((char*)&header)+sizeof(header), buf );

  // read rest of the data
  if ( payloadSize ) {
    if ( fread( buf+sizeof(header), payloadSize, 1, file ) != 1 ) {
      PyObject* fnameObj = PyFile_Name( py_this->m_file );
      if ( feof(file) ) {
        PyErr_Format( PyExc_EOFError, "EOF while reading Dgram payload in file %s", PyString_AsString(fnameObj) ) ;
      } else {
        PyErr_SetFromErrnoWithFilename( PyExc_IOError, PyString_AsString(fnameObj) );
      }

      delete[] buf;
      return 0;
    }
  }

  return pypdsdata::Dgram::PyObject_FromPds( (Pds::Dgram*)buf, 0, dgramSize, ::dgramDtor );
}

}
