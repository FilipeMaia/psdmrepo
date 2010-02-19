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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "EnumType.h"
#include "Exception.h"
#include "types/TypeLib.h"
#include "types/camera/FrameCoord.h"
#include "pdsdata_numpy.h"

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

  PyMethodDef methods[] = {
    {"black_level",       black_level,       METH_NOARGS,  "Returns offset/pedestal setting for camera (before gain)." },
    {"gain_percent",      gain_percent,      METH_NOARGS,  "Returns camera gain setting in percentile [100-3200] = [1x-32x]." },
    {"output_offset",     output_offset,     METH_NOARGS,  "Returns offset/pedestal value in pixel counts." },
    {"output_resolution", output_resolution, METH_NOARGS,  "Returns bit-depth of pixel counts (one of Depth.Eight_bit, Depth.Ten_bit, Depth.Twelve_bit)." },
    {"output_resolution_bits", output_resolution_bits, METH_NOARGS,  "Returns bit-depth of pixel counts (in actual bits)." },
    {"vertical_binning",  vertical_binning,  METH_NOARGS,  "Returns vertical re-binning of output (consecutive rows summed), one of Binning.x1, Binning.x2, etc.." },
    {"output_mirroring",  output_mirroring,  METH_NOARGS,  "Returns geometric transformation of the image." },
    {"vertical_remapping", vertical_remapping, METH_NOARGS, "true: remap the pixels to appear in natural geometric order, false: pixels appear on dual taps from different rows." },
    {"defect_pixel_correction_enabled", defect_pixel_correction_enabled, METH_NOARGS, "If true then correct defective pixels internally." },
    {"output_lookup_table_enabled", output_lookup_table_enabled, METH_NOARGS, "If true then apply output lookup table corrections." },
    {"output_lookup_table", output_lookup_table, METH_NOARGS, "Returns output lookup table: output_value[input_value]." },
    {"number_of_defect_pixels", number_of_defect_pixels, METH_NOARGS, "Returns defective pixel count." },
    {"defect_pixel_coordinates", defect_pixel_coordinates, METH_NOARGS, "Returns list of defective pixel coordinates." },
    {"size",              size,              METH_NOARGS,  "Returns total size of this structure." },
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

}
