#ifndef H5DATATYPES_EVRCONFIGV7_H
#define H5DATATYPES_EVRCONFIGV7_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrConfigV7.
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
#include "pdsdata/evr/ConfigV7.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::EvrData::ConfigV7
//
class EvrConfigV7  {
public:

  typedef Pds::EvrData::ConfigV7 XtcType ;

  EvrConfigV7 () {}
  EvrConfigV7 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc.size() ; }

private:

  uint32_t neventcodes;
  uint32_t npulses;
  uint32_t noutputs;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_EVRCONFIGV7_H
