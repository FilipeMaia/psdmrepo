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
#include "SITConfig/SITConfig.h"

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
#include "ConfigV1.h"
#include "ConfigV2.h"
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../../pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV1, virtual_channel)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV1, lane)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV1, tid)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV1, acq_count)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV1, op_code)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV1, quad)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV1, seq_count)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV1, ticks)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV1, fiducials)
  FUN0_WRAPPER(pypdsdata::CsPad::ElementV1, frame_type)
  PyObject* sb_temp( PyObject* self, PyObject* args );
  PyObject* next( PyObject* self, PyObject* args );
  PyObject* data( PyObject* self, PyObject* args );

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
    {"next",            next,           METH_VARARGS,  "" },
    {"data",            data,           METH_VARARGS,  "" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::CsPad::ElementV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::CsPad::ElementV1::initType( PyObject* module )
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

  BaseType::initType( "ElementV1", module );
}

namespace {
  
PyObject*
sb_temp( PyObject* self, PyObject* args )
{
  const Pds::CsPad::ElementV1* obj = pypdsdata::CsPad::ElementV1::pdsObject( self );
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


// slow bit count
unsigned bitCount(uint32_t mask, unsigned maxBits) {
  unsigned res = 0;
  for (  ; maxBits ; -- maxBits ) {
    if ( mask & 1 ) ++ res ;
    mask >>= 1 ;
  }
  return res ;
}


PyObject*
next( PyObject* self, PyObject* args )
{
  Pds::CsPad::ElementV1* obj = pypdsdata::CsPad::ElementV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* configObj ;
  if ( not PyArg_ParseTuple( args, "O:cspad.ElementV1.next", &configObj ) ) return 0;

  // get segment mask from config object
  uint32_t sMask;
  unsigned payloadSize;
  if ( pypdsdata::CsPad::ConfigV1::Object_TypeCheck( configObj ) ) {
    const Pds::CsPad::ConfigV1* config = pypdsdata::CsPad::ConfigV1::pdsObject( configObj );
    sMask = config->asicMask()==1 ? 0x3 : 0xff;
    payloadSize = config->payloadSize();
  } else if ( pypdsdata::CsPad::ConfigV2::Object_TypeCheck( configObj ) ) {
    const Pds::CsPad::ConfigV2* config = pypdsdata::CsPad::ConfigV2::pdsObject( configObj );
    sMask = config->asicMask()==1 ? 0x3 : 0xff;
    payloadSize = config->payloadSize();
  } else {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a cspad.ConfigV1 or cspad.ConfigV2 object");
    return 0;
  }

  const unsigned nSect = ::bitCount(sMask, Pds::CsPad::ASICsPerQuad/2);
  const unsigned qsize = nSect*Pds::CsPad::ColumnsPerASIC*Pds::CsPad::MaxRowsPerASIC*2;

  // start of pixel data
  const uint16_t* qdata = obj->data();

  // move to next frame
  Pds::CsPad::ElementV1* next = (Pds::CsPad::ElementV1*)(qdata+qsize+2) ;

  // make Python object, parent will be our parent
  pypdsdata::CsPad::ElementV1* py_this = static_cast<pypdsdata::CsPad::ElementV1*>(self);
  return pypdsdata::CsPad::ElementV1::PyObject_FromPds( next, py_this->m_parent, payloadSize, py_this->m_dtor );
}

PyObject*
data( PyObject* self, PyObject* args )
{
  Pds::CsPad::ElementV1* obj = pypdsdata::CsPad::ElementV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* configObj ;
  if ( not PyArg_ParseTuple( args, "O:cspad.ElementV1.data", &configObj ) ) return 0;

  // get segment mask from config object
  uint32_t sMask;
  if ( pypdsdata::CsPad::ConfigV1::Object_TypeCheck( configObj ) ) {
    const Pds::CsPad::ConfigV1* config = pypdsdata::CsPad::ConfigV1::pdsObject( configObj );
    sMask = config->asicMask()==1 ? 0x3 : 0xff;
  } else if ( pypdsdata::CsPad::ConfigV2::Object_TypeCheck( configObj ) ) {
    const Pds::CsPad::ConfigV2* config = pypdsdata::CsPad::ConfigV2::pdsObject( configObj );
    sMask = config->asicMask()==1 ? 0x3 : 0xff;
  } else {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a cspad.ConfigV1 or cspad.ConfigV2 object");
    return 0;
  }
  
  // NumPy type number
  int typenum = NPY_USHORT;

  // not writable
  int flags = NPY_C_CONTIGUOUS ;

  // dimensions
  const unsigned nSect = ::bitCount(sMask, Pds::CsPad::ASICsPerQuad/2);
  npy_intp dims[3] = { nSect, Pds::CsPad::ColumnsPerASIC, Pds::CsPad::MaxRowsPerASIC*2 };

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 3, dims, typenum, 0,
                                (void*)obj->data(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

}
