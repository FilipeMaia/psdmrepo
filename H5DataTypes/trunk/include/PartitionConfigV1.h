#ifndef H5DATATYPES_PARTITIONCONFIGV1_H
#define H5DATATYPES_PARTITIONCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PartitionConfigV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "pdsdata/psddl/partition.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
//  Helper class for Pds::Src class
//
struct PartitionPdsSrc {
public:

  PartitionPdsSrc () {}
  PartitionPdsSrc ( const Pds::Src& src ) : log(src.log()), phy(src.phy()) {}

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  uint32_t log;
  uint32_t phy;
};

//
//  Helper class for Partition::Source class
//
struct PartitionSource {
public:

  PartitionSource () {}
  PartitionSource ( const Pds::Partition::Source& src ) :
    src(src.src()), group(src.group()) {}

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  PartitionPdsSrc src;
  uint32_t group;
};

//
//  Helper class for Pds::Partition::ConfigV1 class
//
class PartitionConfigV1  {
public:

  typedef Pds::Partition::ConfigV1 XtcType ;

  PartitionConfigV1 () {}
  PartitionConfigV1 ( const XtcType& v ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static void store ( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc._sizeof() ; }

protected:

private:

  uint64_t bldMask;
  uint32_t numSources;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_PARTITIONCONFIGV1_H
