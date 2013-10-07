//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class L3TConfigV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/L3TConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/H5DataUtils.h"
#include "hdf5pp/CompoundType.h"

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
L3TConfigV1::L3TConfigV1 (const XtcType& data)
  : module_id_len(data.module_id_len())
  , desc_len(data.desc_len())
  , module_id()
  , description()
{
  module_id = new char[data.module_id_len()+1];
  std::copy(data.module_id(), data.module_id()+data.module_id_len(), module_id);
  module_id[data.module_id_len()] = '\0';

  description = new char[data.desc_len()+1];
  std::copy(data.desc(), data.desc()+data.desc_len(), description);
  description[data.desc_len()] = '\0';
}

L3TConfigV1::~L3TConfigV1()
{
  delete [] module_id;
  delete [] description;
}

hdf5pp::Type
L3TConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
L3TConfigV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<L3TConfigV1>() ;
  type.insert_native<int32_t>("module_id_len", offsetof(L3TConfigV1, module_id_len));
  type.insert_native<int32_t>("desc_len", offsetof(L3TConfigV1, desc_len));
  type.insert_native<const char*>("module_id", offsetof(L3TConfigV1, module_id));
  type.insert_native<const char*>("desc", offsetof(L3TConfigV1, description));

  return type ;
}

void
L3TConfigV1::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  L3TConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
