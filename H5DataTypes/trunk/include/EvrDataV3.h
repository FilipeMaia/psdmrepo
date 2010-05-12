#ifndef H5DATATYPES_EVRDATAV3_H
#define H5DATATYPES_EVRDATAV3_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrDataV3.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "hdf5pp/Group.h"
#include "pdsdata/evr/DataV3.hh"
#include "pdsdata/evr/ConfigV3.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

struct EvrDataV3_Data {
  uint32_t numFifoEvents;
};

class EvrDataV3  {
public:

  typedef Pds::EvrData::DataV3 XtcType ;
  typedef Pds::EvrData::ConfigV3 ConfigXtcType ;

  // Default constructor
  EvrDataV3 () {}
  EvrDataV3 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static hdf5pp::Type stored_fifoevent_type( const ConfigXtcType& config ) ;

protected:

private:

  EvrDataV3_Data m_data ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_EVRDATAV3_H
