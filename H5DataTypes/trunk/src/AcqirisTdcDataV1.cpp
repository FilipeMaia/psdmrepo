//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisTdcDataV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/AcqirisTdcDataV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/VlenType.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

//----------------
// Constructors --
//----------------
AcqirisTdcDataV1::AcqirisTdcDataV1 ()
{
}

AcqirisTdcDataV1::AcqirisTdcDataV1 (size_t size, const XtcType* xtcData)
  : m_size(size)
  , m_data(new AcqirisTdcDataV1_Data[size])
{

  for (size_t i = 0 ; i != size ; ++ i ) {
    m_data[i].source = xtcData[i].source();
    const class XtcType::Common& com = static_cast<const class XtcType::Common&>(xtcData[i]);
    m_data[i].overflow = com.overflow();
    m_data[i].value = com.nhits();
  }
}

AcqirisTdcDataV1::~AcqirisTdcDataV1 ()
{
  delete [] m_data;
}

hdf5pp::Type
AcqirisTdcDataV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
AcqirisTdcDataV1::native_type()
{
  hdf5pp::EnumType<uint8_t> srcType = hdf5pp::EnumType<uint8_t>::enumType();
  srcType.insert("Common", XtcType::Common);
  srcType.insert("Chan1", XtcType::Chan1);
  srcType.insert("Chan2", XtcType::Chan2);
  srcType.insert("Chan3", XtcType::Chan3);
  srcType.insert("Chan4", XtcType::Chan4);
  srcType.insert("Chan5", XtcType::Chan5);
  srcType.insert("Chan6", XtcType::Chan6);
  srcType.insert("AuxIO", XtcType::AuxIO);

  hdf5pp::CompoundType baseType = hdf5pp::CompoundType::compoundType<AcqirisTdcDataV1_Data>() ;
  baseType.insert( "source", offsetof(AcqirisTdcDataV1_Data, source), srcType ) ;
  baseType.insert_native<uint8_t>( "overflow", offsetof(AcqirisTdcDataV1_Data, overflow) ) ;
  baseType.insert_native<uint32_t>( "value", offsetof(AcqirisTdcDataV1_Data, value) ) ;

  hdf5pp::Type type = hdf5pp::VlenType::vlenType ( baseType );

  return type ;
}

} // namespace H5DataTypes
