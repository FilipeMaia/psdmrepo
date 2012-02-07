//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ElementV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ElementV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../../pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ElementV1, virtual_channel)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ElementV1, lane)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ElementV1, tid)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ElementV1, acq_count)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ElementV1, op_code)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ElementV1, quad)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ElementV1, seq_count)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ElementV1, ticks)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ElementV1, fiducials)
  FUN0_WRAPPER(pypdsdata::CsPad2x2::ElementV1, frame_type)
  PyObject* sb_temp( PyObject* self, PyObject* args );
  PyObject* data( PyObject* self, PyObject* );
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"virtual_channel", virtual_channel, METH_NOARGS,  "" },
    {"lane",            lane,           METH_NOARGS,  "" },
    {"tid",             tid,            METH_NOARGS,  "" },
    {"acq_count",       acq_count,      METH_NOARGS,  "" },
    {"op_code",         op_code,        METH_NOARGS,  "" },
    {"quad",            quad,           METH_NOARGS,  "" },
    {"seq_count",       seq_count,      METH_NOARGS,  "" },
    {"ticks",           ticks,          METH_NOARGS,  "" },
    {"fiducials",       fiducials,      METH_NOARGS,  "" },
    {"frame_type",      frame_type,     METH_NOARGS,  "" },
    {"sb_temp",         sb_temp,        METH_VARARGS,  "" },
    {"data",            data,           METH_NOARGS,  "" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::CsPad2x2::ElementV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::CsPad2x2::ElementV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // add an enum analog to this class 
  type->tp_dict = PyDict_New();
  PyObject* val = PyInt_FromLong(Pds::CsPad2x2::ColumnsPerASIC);
  PyDict_SetItemString( type->tp_dict, "ColumnsPerASIC", val );
  Py_CLEAR(val);
  val = PyInt_FromLong(Pds::CsPad2x2::MaxRowsPerASIC);
  PyDict_SetItemString( type->tp_dict, "MaxRowsPerASIC", val );
  Py_CLEAR(val);

  BaseType::initType( "ElementV1", module );
}

namespace {
  
PyObject*
sb_temp( PyObject* self, PyObject* args )
{
  const Pds::CsPad2x2::ElementV1* obj = pypdsdata::CsPad2x2::ElementV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned index ;
  if ( not PyArg_ParseTuple( args, "I:ElementV1.sb_temp", &index ) ) return 0;

  if ( index >= 4 ) {
    PyErr_SetString(PyExc_IndexError, "index outside of range [0..3] in ElementV1.sb_temp()");
    return 0;
  }
  
  return PyInt_FromLong( obj->sb_temp(index) );
}


PyObject*
data( PyObject* self, PyObject* )
{
  Pds::CsPad2x2::ElementV1* obj = pypdsdata::CsPad2x2::ElementV1::pdsObject( self );
  if ( not obj ) return 0;

  // NumPy type number
  int typenum = NPY_USHORT;

  // not writable
  int flags = NPY_C_CONTIGUOUS ;

  // dimensions
  const unsigned nSect = 2;
  npy_intp dims[3] = { Pds::CsPad2x2::ColumnsPerASIC, Pds::CsPad2x2::MaxRowsPerASIC*2, nSect };

  // start of pixel data
  uint16_t* qdata = &obj->pair[0][0].s0;

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 3, dims, typenum, 0,
                                qdata, 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

PyObject*
_repr( PyObject *self )
{
  Pds::CsPad2x2::ElementV1* obj = pypdsdata::CsPad2x2::ElementV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "CsPad2x2.ElementV1(quad=" << obj->quad()
      << ", virtual_channel=" << obj->virtual_channel()
      << ", lane=" << obj->lane()
      << ", tid=" << obj->tid()
      << ", acq_count=" << obj->acq_count()
      << ", op_code=" << obj->op_code()
      << ", seq_count=" << obj->seq_count()
      << ", ticks=" << obj->ticks()
      << ", fiducials=" << obj->fiducials()
      << ", frame_type=" << obj->frame_type()
      << ")";

  return PyString_FromString( str.str().c_str() );
}

}
