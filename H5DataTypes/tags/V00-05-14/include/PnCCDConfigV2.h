#ifndef H5DATATYPES_PNCCDCONFIGV2_H
#define H5DATATYPES_PNCCDCONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnCCDConfigV2.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "pdsdata/psddl/pnccd.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::PNCCD::ConfigV2
//
class PnCCDConfigV2 : boost::noncopyable {
public:

  typedef Pds::PNCCD::ConfigV2 XtcType ;

  PnCCDConfigV2() {}
  PnCCDConfigV2 ( const XtcType& config ) ;

  ~PnCCDConfigV2();

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static void store ( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc._sizeof() ; }

private:

  uint32_t numLinks;
  uint32_t payloadSizePerLink;
  uint32_t numChannels;
  uint32_t numRows;
  uint32_t numSubmoduleChannels;
  uint32_t numSubmoduleRows;
  uint32_t numSubmodules;
  uint32_t camexMagic;
  char*    info;
  char*    timingFName;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_PNCCDCONFIGV2_H
