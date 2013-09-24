//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ElementV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ElementV2.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ConfigV2.h"
#include "ConfigV3.h"
#include "ConfigV4.h"
#include "ConfigV5.h"
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../../pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV2, virtual_channel)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV2, lane)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV2, tid)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV2, acq_count)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV2, op_code)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV2, quad)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV2, seq_count)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV2, ticks)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV2, fiducials)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV2, frame_type)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV2, sb_temp)
  PyObject* data( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"virtual_channel", virtual_channel, METH_NOARGS,  "self.virtual_channel() -> int\n\nReturns integer number" },
    {"lane",            lane,           METH_NOARGS,  "self.lane() -> int\n\nReturns integer number" },
    {"tid",             tid,            METH_NOARGS,  "self.tid() -> int\n\nReturns integer number" },
    {"acq_count",       acq_count,      METH_NOARGS,  "self.acq_count() -> int\n\nReturns integer number" },
    {"op_code",         op_code,        METH_NOARGS,  "self.op_code() -> int\n\nReturns integer number" },
    {"quad",            quad,           METH_NOARGS,  "self.quad() -> int\n\nReturns quadrant number" },
    {"seq_count",       seq_count,      METH_NOARGS,  "self.seq_count() -> int\n\nReturns sequence counter" },
    {"ticks",           ticks,          METH_NOARGS,  "self.ticks() -> int\n\nReturns integer number" },
    {"fiducials",       fiducials,      METH_NOARGS,  "self.fiducials() -> int\n\nReturns integer number" },
    {"frame_type",      frame_type,     METH_NOARGS,  "self.frame_type() -> int\n\nReturns integer number" },
    {"sb_temp",         sb_temp,        METH_NOARGS,  "self.sb_temp() -> list\n\nRetuns list of integers" },
    {"data",            data,           METH_VARARGS,  "self.data(cfg: ConfigV*) -> numpy.ndarray\n\nReturns data array for this quadrant" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::CsPad::ElementV2 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::CsPad::ElementV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // add an enum analog to this class 
  type->tp_dict = PyDict_New();
  PyObject* val = PyInt_FromLong(Pds::CsPad::ColumnsPerASIC);
  PyDict_SetItemString( type->tp_dict, "ColumnsPerASIC", val );
  Py_CLEAR(val);
  val = PyInt_FromLong(Pds::CsPad::MaxRowsPerASIC);
  PyDict_SetItemString( type->tp_dict, "MaxRowsPerASIC", val );
  Py_CLEAR(val);

  BaseType::initType( "ElementV2", module );
}

namespace {
  
PyObject*
data( PyObject* self, PyObject* args )
{
  Pds::CsPad::ElementV2* obj = pypdsdata::CsPad::ElementV2::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* configObj ;
  if ( not PyArg_ParseTuple( args, "O:cspad.ElementV2.data", &configObj ) ) return 0;

  // get image data
  ndarray<const int16_t, 3> data;
  if ( pypdsdata::CsPad::ConfigV2::Object_TypeCheck( configObj ) ) {
    const Pds::CsPad::ConfigV2* config = pypdsdata::CsPad::ConfigV2::pdsObject( configObj );
    data = obj->data(*config);
  } else if ( pypdsdata::CsPad::ConfigV3::Object_TypeCheck( configObj ) ) {
    const Pds::CsPad::ConfigV3* config = pypdsdata::CsPad::ConfigV3::pdsObject( configObj );
    data = obj->data(*config);
  } else if ( pypdsdata::CsPad::ConfigV4::Object_TypeCheck( configObj ) ) {
    const Pds::CsPad::ConfigV4* config = pypdsdata::CsPad::ConfigV4::pdsObject( configObj );
    data = obj->data(*config);
  } else if ( pypdsdata::CsPad::ConfigV5::Object_TypeCheck( configObj ) ) {
    const Pds::CsPad::ConfigV5* config = pypdsdata::CsPad::ConfigV5::pdsObject( configObj );
    data = obj->data(*config);
  } else {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a cspad.ConfigV* object");
    return 0;
  }
  
  // NumPy type number
  int typenum = NPY_USHORT;

  // not writable
  int flags = NPY_C_CONTIGUOUS ;

  // dimensions
  npy_intp dims[3] = { data.shape()[0], data.shape()[1], data.shape()[2] };

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 3, dims, typenum, 0,
                                (void*)data.data(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}


}
