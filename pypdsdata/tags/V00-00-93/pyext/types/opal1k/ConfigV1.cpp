//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../EnumType.h"
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../camera/FrameCoord.h"
#include "../../pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  pypdsdata::EnumType::Enum depthEnumValues[] = {
      { "Eight_bit",  Pds::Opal1k::ConfigV1::Eight_bit },
      { "Ten_bit",    Pds::Opal1k::ConfigV1::Ten_bit },
      { "Twelve_bit", Pds::Opal1k::ConfigV1::Twelve_bit },
      { 0, 0 }
  };
  pypdsdata::EnumType depthEnum ( "Depth", depthEnumValues );

  pypdsdata::EnumType::Enum binningEnumValues[] = {
      { "x1", Pds::Opal1k::ConfigV1::x1 },
      { "x2", Pds::Opal1k::ConfigV1::x2 },
      { "x4", Pds::Opal1k::ConfigV1::x4 },
      { "x8", Pds::Opal1k::ConfigV1::x8 },
      { 0, 0 }
  };
  pypdsdata::EnumType binningEnum ( "Binning", binningEnumValues );

  pypdsdata::EnumType::Enum mirroringEnumValues[] = {
      { "None",   Pds::Opal1k::ConfigV1::None },
      { "HFlip",  Pds::Opal1k::ConfigV1::HFlip },
      { "VFlip",  Pds::Opal1k::ConfigV1::VFlip },
      { "HVFlip", Pds::Opal1k::ConfigV1::HVFlip },
      { 0, 0 }
  };
  pypdsdata::EnumType mirroringEnum ( "Mirroring", mirroringEnumValues );


  // list of enums
  pypdsdata::TypeLib::EnumEntry enums[] = {
        { "LUT_Size", Pds::Opal1k::ConfigV1::LUT_Size },

        { "Row_Pixels", Pds::Opal1k::ConfigV1::Row_Pixels },

        { "Column_Pixels", Pds::Opal1k::ConfigV1::Column_Pixels },

        { 0, 0 }
  };

  // methods
  FUN0_WRAPPER(pypdsdata::Opal1k::ConfigV1, black_level)
  FUN0_WRAPPER(pypdsdata::Opal1k::ConfigV1, gain_percent)
  FUN0_WRAPPER(pypdsdata::Opal1k::ConfigV1, output_offset)
  ENUM_FUN0_WRAPPER(pypdsdata::Opal1k::ConfigV1, output_resolution, depthEnum)
  FUN0_WRAPPER(pypdsdata::Opal1k::ConfigV1, output_resolution_bits)
  ENUM_FUN0_WRAPPER(pypdsdata::Opal1k::ConfigV1, vertical_binning, binningEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Opal1k::ConfigV1, output_mirroring, mirroringEnum)
  FUN0_WRAPPER(pypdsdata::Opal1k::ConfigV1, vertical_remapping)
  FUN0_WRAPPER(pypdsdata::Opal1k::ConfigV1, defect_pixel_correction_enabled)
  FUN0_WRAPPER(pypdsdata::Opal1k::ConfigV1, output_lookup_table_enabled)
  FUN0_WRAPPER(pypdsdata::Opal1k::ConfigV1, number_of_defect_pixels)
  FUN0_WRAPPER(pypdsdata::Opal1k::ConfigV1, size)
  PyObject* output_lookup_table( PyObject* self, PyObject* );
  PyObject* defect_pixel_coordinates( PyObject* self, PyObject* );
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"black_level",       black_level,       METH_NOARGS,  
        "self.black_level() -> int\n\nReturns offset/pedestal setting for camera (before gain)." },
    {"gain_percent",      gain_percent,      METH_NOARGS,  
        "self.gain_percent() -> int\n\nReturns camera gain setting in percentile [100-3200] = [1x-32x]." },
    {"output_offset",     output_offset,     METH_NOARGS,  
        "self.output_offset() -> int\n\nReturns offset/pedestal value in pixel counts." },
    {"output_resolution", output_resolution, METH_NOARGS,  
        "self.output_resolution() -> Depth enum\n\nReturns bit-depth of pixel counts (one of :py:class:`Depth` enums)." },
    {"output_resolution_bits", output_resolution_bits, METH_NOARGS,  
        "self.output_resolution_bits() -> int\n\nReturns bit-depth of pixel counts (in actual bits)." },
    {"vertical_binning",  vertical_binning,  METH_NOARGS,  
        "self.vertical_binning() -> Binning enum\n\nReturns vertical re-binning of output (consecutive rows summed), one of :py:class:`Binning` enums" },
    {"output_mirroring",  output_mirroring,  METH_NOARGS,  
        "self.output_mirroring() -> Mirroring enum\n\nReturns geometric transformation of the image, one of :py:class:`Mirroring` enums" },
    {"vertical_remapping", vertical_remapping, METH_NOARGS, 
        "self.vertical_remapping() -> bool\n\ntrue: remap the pixels to appear in natural geometric order, false: pixels appear on dual taps from different rows." },
    {"defect_pixel_correction_enabled", defect_pixel_correction_enabled, METH_NOARGS, 
        "self.defect_pixel_correction_enabled() -> bool\n\nIf true then correct defective pixels internally." },
    {"output_lookup_table_enabled", output_lookup_table_enabled, METH_NOARGS, 
        "self.output_lookup_table_enabled() -> bool\n\nIf true then apply output lookup table corrections." },
    {"output_lookup_table", output_lookup_table, METH_NOARGS, 
        "self.output_lookup_table() -> numpy.ndarray\n\nReturns output lookup table: output_value[input_value]." },
    {"number_of_defect_pixels", number_of_defect_pixels, METH_NOARGS, 
        "self.number_of_defect_pixels() -> int\n\nReturns defective pixel count." },
    {"defect_pixel_coordinates", defect_pixel_coordinates, METH_NOARGS, 
        "self.defect_pixel_coordinates() -> list of camera.FrameCoord\n\nReturns list of defective pixel coordinates (:py:class:`_pdsdata.camera.FrameCoord` objects)." },
    {"size",              size,              METH_NOARGS,  "self.size() -> int\n\nReturns total size of this structure." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Opal1k::ConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Opal1k::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Depth", depthEnum.type() );
  PyDict_SetItemString( tp_dict, "Binning", binningEnum.type() );
  PyDict_SetItemString( tp_dict, "Mirroring", mirroringEnum.type() );
  pypdsdata::TypeLib::DefineEnums( tp_dict, ::enums );
  type->tp_dict = tp_dict;

  BaseType::initType( "ConfigV1", module );
}

namespace {

PyObject*
output_lookup_table( PyObject* self, PyObject* )
{
  const Pds::Opal1k::ConfigV1* obj = pypdsdata::Opal1k::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  // NumPy type number
  int typenum = NPY_USHORT;

  // not writable
  int flags = NPY_C_CONTIGUOUS ;

  // dimensions
  npy_intp dims[1] = { obj->output_lookup_table_enabled() ? Pds::Opal1k::ConfigV1::LUT_Size : 0 };

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 1, dims, typenum, 0,
                                (void*)obj->output_lookup_table(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

PyObject*
defect_pixel_coordinates( PyObject* self, PyObject* )
{
  const Pds::Opal1k::ConfigV1* obj = pypdsdata::Opal1k::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  unsigned size = obj->number_of_defect_pixels() ;
  PyObject* list = PyList_New(size);

  // copy coordinates to the list
  const Pds::Camera::FrameCoord* coords = obj->defect_pixel_coordinates();
  for ( unsigned i = 0; i < size; ++ i ) {
    PyObject* obj = pypdsdata::Camera::FrameCoord::PyObject_FromPds(coords[i]) ;
    PyList_SET_ITEM( list, i, obj );
  }

  return list;
}

PyObject*
_repr( PyObject *self )
{
  Pds::Opal1k::ConfigV1* obj = pypdsdata::Opal1k::ConfigV1::pdsObject(self);
  if (not obj) return 0;

  std::ostringstream str ;
  str << "opal1k.ConfigV1(black_level=" << obj->black_level()
      << ", gain_percent=" << obj->gain_percent() 
      << ", output_offset=" << obj->output_offset() 
      << ", output_bits=" << obj->output_resolution_bits()
      << ", ...)";
  return PyString_FromString( str.str().c_str() );
}

}
