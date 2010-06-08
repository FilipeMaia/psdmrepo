#ifndef H5DATATYPES_EVRCONFIGV4_H
#define H5DATATYPES_EVRCONFIGV4_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrConfigV4.
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
#include "pdsdata/evr/ConfigV4.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::EvrData::ConfigV4
//
struct EvrConfigV4_Data {
  uint32_t neventcodes;
  uint32_t npulses;
  uint32_t noutputs;
};

class EvrConfigV4  {
public:

  typedef Pds::EvrData::ConfigV4 XtcType ;

  EvrConfigV4 () {}
  EvrConfigV4 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc.size() ; }

private:

  EvrConfigV4_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_EVRCONFIGV4_H
