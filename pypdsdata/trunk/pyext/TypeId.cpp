//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TypeId...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TypeId.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <new>
#include <cstdio>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "EnumType.h"
#include "types/TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  pypdsdata::EnumType::Enum typeEnumValues[] = {
      { "Any",                Pds::TypeId::Any },
      { "Id_Xtc",             Pds::TypeId::Id_Xtc },
      { "Id_Frame",           Pds::TypeId::Id_Frame },
      { "Id_AcqWaveform",     Pds::TypeId::Id_AcqWaveform },
      { "Id_AcqConfig",       Pds::TypeId::Id_AcqConfig },
      { "Id_TwoDGaussian",    Pds::TypeId::Id_TwoDGaussian },
      { "Id_Opal1kConfig",    Pds::TypeId::Id_Opal1kConfig },
      { "Id_FrameFexConfig",  Pds::TypeId::Id_FrameFexConfig },
      { "Id_EvrConfig",       Pds::TypeId::Id_EvrConfig },
      { "Id_TM6740Config",    Pds::TypeId::Id_TM6740Config },
      { "Id_ControlConfig",   Pds::TypeId::Id_ControlConfig },
      { "Id_pnCCDframe",      Pds::TypeId::Id_pnCCDframe },
      { "Id_pnCCDconfig",     Pds::TypeId::Id_pnCCDconfig },
      { "Id_Epics",           Pds::TypeId::Id_Epics },
      { "Id_FEEGasDetEnergy", Pds::TypeId::Id_FEEGasDetEnergy },
      { "Id_EBeam",           Pds::TypeId::Id_EBeam },
      { "Id_PhaseCavity",     Pds::TypeId::Id_PhaseCavity },
      { "Id_PrincetonFrame",  Pds::TypeId::Id_PrincetonFrame },
      { "Id_PrincetonConfig", Pds::TypeId::Id_PrincetonConfig },
      { "Id_EvrData",         Pds::TypeId::Id_EvrData },
      { "Id_FrameFccdConfig", Pds::TypeId::Id_FrameFccdConfig },
      { "Id_FccdConfig",      Pds::TypeId::Id_FccdConfig },
      { "Id_IpimbData",       Pds::TypeId::Id_IpimbData },
      { "Id_IpimbConfig",     Pds::TypeId::Id_IpimbConfig },
      { "Id_EncoderData",     Pds::TypeId::Id_EncoderData },
      { "Id_EncoderConfig",   Pds::TypeId::Id_EncoderConfig },
      { "Id_EvrIOConfig",     Pds::TypeId::Id_EvrIOConfig },
      { "Id_PrincetonInfo",   Pds::TypeId::Id_PrincetonInfo },
      { "Id_CspadElement",    Pds::TypeId::Id_CspadElement },
      { "Id_CspadConfig",     Pds::TypeId::Id_CspadConfig },
      { "Id_IpmFexConfig",    Pds::TypeId::Id_IpmFexConfig },
      { "Id_IpmFex",          Pds::TypeId::Id_IpmFex },
      { "Id_DiodeFexConfig",  Pds::TypeId::Id_DiodeFexConfig },
      { "Id_DiodeFex",        Pds::TypeId::Id_DiodeFex },
      { "Id_PimImageConfig",  Pds::TypeId::Id_PimImageConfig },
      { "Id_SharedIpimb",     Pds::TypeId::Id_SharedIpimb },
      { "Id_AcqTdcConfig",    Pds::TypeId::Id_AcqTdcConfig },
      { "Id_AcqTdcData",      Pds::TypeId::Id_AcqTdcData },
      { "Id_Index",           Pds::TypeId::Id_Index },
      { "Id_XampsConfig",     Pds::TypeId::Id_XampsConfig },
      { "Id_XampsElement",    Pds::TypeId::Id_XampsElement },
      { "Id_Cspad2x2Element", Pds::TypeId::Id_Cspad2x2Element },
      { "Id_SharedPim",       Pds::TypeId::Id_SharedPim },
      { "Id_Cspad2x2Config",  Pds::TypeId::Id_Cspad2x2Config },
      { "Id_FexampConfig",    Pds::TypeId::Id_FexampConfig },
      { "Id_FexampElement",   Pds::TypeId::Id_FexampElement },
      { "Id_Gsc16aiConfig",   Pds::TypeId::Id_Gsc16aiConfig },
      { "Id_Gsc16aiData",     Pds::TypeId::Id_Gsc16aiData },
      { "Id_PhasicsConfig",   Pds::TypeId::Id_PhasicsConfig },
      { "Id_TimepixConfig",   Pds::TypeId::Id_TimepixConfig },
      { "Id_TimepixData",     Pds::TypeId::Id_TimepixData },
      { "Id_CspadCompressedElement", Pds::TypeId::Id_CspadCompressedElement },
      { "Id_OceanOpticsConfig", Pds::TypeId::Id_OceanOpticsConfig },
      { "Id_OceanOpticsData", Pds::TypeId::Id_OceanOpticsData },
      { "Id_EpicsConfig",     Pds::TypeId::Id_EpicsConfig },
      { "Id_FliConfig",       Pds::TypeId::Id_FliConfig },
      { "Id_FliFrame",        Pds::TypeId::Id_FliFrame },
      { "Id_QuartzConfig",    Pds::TypeId::Id_QuartzConfig },
      { "Id_AndorConfig",     Pds::TypeId::Id_AndorConfig },
      { "Id_AndorFrame",      Pds::TypeId::Id_AndorFrame },
      { "Id_UsdUsbData",      Pds::TypeId::Id_UsdUsbData },
      { "Id_UsdUsbConfig",    Pds::TypeId::Id_UsdUsbConfig },
      { "Id_GMD",             Pds::TypeId::Id_GMD },
      { "Id_SharedAcqADC",    Pds::TypeId::Id_SharedAcqADC },
      { "Id_OrcaConfig",      Pds::TypeId::Id_OrcaConfig },
      { "Id_ImpData",         Pds::TypeId::Id_ImpData },
      { "Id_ImpConfig",       Pds::TypeId::Id_ImpConfig },
      { "Id_AliasConfig",     Pds::TypeId::Id_AliasConfig },
      { "Id_L3TConfig",       Pds::TypeId::Id_L3TConfig },
      { "Id_L3TData",         Pds::TypeId::Id_L3TData },
      { "Id_Spectrometer",    Pds::TypeId::Id_Spectrometer },
      { "Id_RayonixConfig",   Pds::TypeId::Id_RayonixConfig },
      { "Id_EpixConfig",      Pds::TypeId::Id_EpixConfig },
      { "Id_EpixElement",     Pds::TypeId::Id_EpixElement },
      { "Id_EpixSamplerConfig", Pds::TypeId::Id_EpixSamplerConfig },
      { "Id_EpixSamplerElement", Pds::TypeId::Id_EpixSamplerElement },
      { "Id_EvsConfig",       Pds::TypeId::Id_EvsConfig },
      { "Id_PartitionConfig", Pds::TypeId::Id_PartitionConfig },
      { "Id_PimaxConfig",     Pds::TypeId::Id_PimaxConfig },
      { "Id_PimaxFrame",      Pds::TypeId::Id_PimaxFrame },
      { "Id_Arraychar",       Pds::TypeId::Id_Arraychar },
      { "NumberOf",           Pds::TypeId::NumberOf },
      { 0, 0 }
  };
  pypdsdata::EnumType typeEnum ( "Type", typeEnumValues );

  // standard Python stuff
  int TypeId_init( PyObject* self, PyObject* args, PyObject* kwds );
  PyObject* TypeId_repr( PyObject* self );
  long TypeId_hash( PyObject* self );
  int TypeId_compare( PyObject *self, PyObject *other);

  // type-specific methods
  FUN0_WRAPPER_EMBEDDED(pypdsdata::TypeId, value);
  ENUM_FUN0_WRAPPER_EMBEDDED(pypdsdata::TypeId, id, typeEnum);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::TypeId, version);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::TypeId, compressed);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::TypeId, compressed_version);

  PyMethodDef methods[] = {
    { "value",    value,   METH_NOARGS, "self.value() -> int\n\nReturns the whole type ID number including version" },
    { "id",       id,      METH_NOARGS, "self.id() -> Type\n\nReturns the type ID number without version (:py:class:`Type`)" },
    { "version",  version, METH_NOARGS, "self.version() -> int\n\nReturns the type ID version number" },
    { "compressed", compressed, METH_NOARGS, "self.compressed() -> bool\n\nReturns true if the object is compressed" },
    { "compressed_version", compressed_version, METH_NOARGS, "self.compressed_version() -> int\n\nReturns the type ID version number excluding compressed bits" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::TypeId class.\n\n"
      "This class inherits from a int type, the instances of this class\n"
      "are regular numbers with some additional niceties: repr() and str()\n"
      "functions will print string representaion of the enum values.\n"
      "Class defines several attributes which correspond to the C++ enum values.\n\n"
      "Class constructor takes three optional positional arguments - type id,\n"
      "version number, and compressed flag. If missing the values are initialized \n"
      "with 0.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::TypeId::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_init = TypeId_init;
  type->tp_hash = TypeId_hash;
  type->tp_compare = TypeId_compare;
  type->tp_repr = TypeId_repr;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "Type", typeEnum.type() );
  PyObject* val = PyInt_FromLong(Pds::TypeId::VCompressed);
  PyDict_SetItemString( type->tp_dict, "VCompressed", val );
  Py_XDECREF(val);

  BaseType::initType( "TypeId", module );
}

void 
pypdsdata::TypeId::print(std::ostream& out) const
{
  if ( m_obj.compressed() ) {
    out << Pds::TypeId::name(m_obj.id()) << "_V" << m_obj.compressed_version() << "/compressed";
  } else if ( m_obj.version() ) {
    out << Pds::TypeId::name(m_obj.id()) << "_V" << m_obj.compressed_version();
  } else {
    out << Pds::TypeId::name(m_obj.id());
  }
}

namespace {

int
TypeId_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::TypeId* py_this = (pypdsdata::TypeId*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  unsigned val = Pds::TypeId::Any;
  unsigned version = 0;
  int compressed = 0;
  if ( not PyArg_ParseTuple( args, "|IIi:TypeId", &val, &version, &compressed ) ) return -1;

  if ( val >= Pds::TypeId::NumberOf ) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "Error: TypeId out of range: %d (range is [0..%d])", val, Pds::TypeId::NumberOf-1);
    if (PyErr_WarnEx(PyExc_RuntimeWarning, buf, 3) < 0) {
      return -1;
    }
  }

  new(&py_this->m_obj) Pds::TypeId( Pds::TypeId::Type(val), version, compressed );

  return 0;
}

long
TypeId_hash( PyObject* self )
{
  pypdsdata::TypeId* py_this = (pypdsdata::TypeId*) self;
  long hash = py_this->m_obj.value();
  return hash;
}

int
TypeId_compare( PyObject* self, PyObject* other )
{
  pypdsdata::TypeId* py_this = (pypdsdata::TypeId*) self;
  pypdsdata::TypeId* py_other = (pypdsdata::TypeId*) other;
  if ( py_this->m_obj.value() > py_other->m_obj.value() ) return 1 ;
  if ( py_this->m_obj.value() == py_other->m_obj.value() ) return 0 ;
  return -1 ;
}

PyObject*
TypeId_repr( PyObject* self )
{
  pypdsdata::TypeId* py_this = (pypdsdata::TypeId*) self;
  return PyString_FromFormat("<TypeId(%s,%d)>", Pds::TypeId::name(py_this->m_obj.id()),
      py_this->m_obj.compressed_version() );
}

}
