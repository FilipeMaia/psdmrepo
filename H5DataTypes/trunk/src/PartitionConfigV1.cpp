//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PartitionConfigV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/PartitionConfigV1.h"

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
PartitionPdsSrc::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
PartitionPdsSrc::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<PartitionPdsSrc>() ;
  type.insert_native<uint32_t>( "log", offsetof(PartitionPdsSrc, log) ) ;
  type.insert_native<uint32_t>( "phy", offsetof(PartitionPdsSrc, phy) ) ;
  return type;
}



hdf5pp::Type
PartitionSource::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
PartitionSource::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<PartitionSource>() ;
  type.insert( "src", offsetof(PartitionSource, src), PartitionPdsSrc::native_type() ) ;
  type.insert_native<uint32_t>( "group", offsetof(PartitionSource, group) ) ;
  return type;
}


PartitionConfigV1::PartitionConfigV1(const XtcType& data)
  : bldMask(data.bldMask())
  , numSources(data.numSources())
{
}

hdf5pp::Type
PartitionConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
PartitionConfigV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<PartitionConfigV1>() ;
  type.insert_native<uint64_t>( "bldMask", offsetof(PartitionConfigV1, bldMask) ) ;
  type.insert_native<uint32_t>( "numSources", offsetof(PartitionConfigV1, numSources) ) ;
  return type;
}

void
PartitionConfigV1::store(const XtcType& config, hdf5pp::Group grp)
{
  // make scalar data set for main object
  PartitionConfigV1 data(config);
  storeDataObject ( data, "config", grp ) ;

  // make array data set for subobject
  const ndarray<const Pds::Partition::Source, 1>& pdsdata = config.sources();
  const uint32_t count = pdsdata.size();
  PartitionSource adata[count];
  for ( uint32_t i = 0 ; i < count ; ++ i ) {
    adata[i] = PartitionSource(pdsdata[i]);
  }
  storeDataObjects ( count, adata, "sources", grp ) ;
}

} // namespace H5DataTypes
