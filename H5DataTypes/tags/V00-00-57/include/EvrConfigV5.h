#ifndef H5DATATYPES_EVRCONFIGV5_H
#define H5DATATYPES_EVRCONFIGV5_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrConfigV5.
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
#include "pdsdata/evr/ConfigV5.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::EvrData::ConfigV5
//
struct EvrConfigV5_Data {
  uint32_t neventcodes;
  uint32_t npulses;
  uint32_t noutputs;
  
};

class EvrConfigV5  {
public:

  typedef Pds::EvrData::ConfigV5 XtcType ;

  EvrConfigV5 () {}
  EvrConfigV5 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc.size() ; }

private:

  EvrConfigV5_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_EVRCONFIGV5_H
