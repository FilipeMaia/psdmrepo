//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AliasConfigV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/AliasConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <cstring>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/H5DataUtils.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {


hdf5pp::Type
AliasPdsSrc::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
AliasPdsSrc::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<AliasPdsSrc>() ;
  type.insert_native<uint32_t>( "log", offsetof(AliasPdsSrc, log) ) ;
  type.insert_native<uint32_t>( "phy", offsetof(AliasPdsSrc, phy) ) ;
  return type;
}


AliasSrcAlias::AliasSrcAlias(const Pds::Alias::SrcAlias& srcAlias)
  : src(srcAlias.src())
{
  std::fill_n(aliasName, int(AliasNameMax), '\0');
  std::strncpy(aliasName, srcAlias.aliasName(), AliasNameMax-1);
}

hdf5pp::Type
AliasSrcAlias::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
AliasSrcAlias::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<AliasSrcAlias>() ;
  type.insert( "src", offsetof(AliasSrcAlias, src), AliasPdsSrc::native_type() ) ;
  type.insert_native<const char*>( "aliasName", offsetof(AliasSrcAlias, aliasName), AliasNameMax ) ;
  return type;
}


AliasConfigV1::AliasConfigV1(const XtcType& data)
  : numSrcAlias(data.numSrcAlias())
{
}

hdf5pp::Type
AliasConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
AliasConfigV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<AliasConfigV1>() ;
  type.insert_native<uint32_t>( "numSrcAlias", offsetof(AliasConfigV1, numSrcAlias) ) ;
  return type;
}

void
AliasConfigV1::store(const XtcType& config, hdf5pp::Group grp)
{
  // make scalar data set for main object
  AliasConfigV1 data(config);
  storeDataObject ( data, "config", grp ) ;

  // make array data set for subobject
  const ndarray<const Pds::Alias::SrcAlias, 1>& pdsdata = config.srcAlias();
  const uint32_t count = pdsdata.size();
  AliasSrcAlias adata[count];
  for ( uint32_t i = 0 ; i < count ; ++ i ) {
    adata[i] = AliasSrcAlias(pdsdata[i]);
  }
  storeDataObjects ( count, adata, "aliases", grp ) ;
}

} // namespace H5DataTypes
