#ifndef H5DATATYPES_EVRCONFIGV3_H
#define H5DATATYPES_EVRCONFIGV3_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrConfigV3.
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
#include "pdsdata/psddl/evr.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::EvrData::ConfigV3
//
class EvrConfigV3  {
public:

  typedef Pds::EvrData::ConfigV3 XtcType ;

  EvrConfigV3 () {}
  EvrConfigV3 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc._sizeof() ; }

private:

  uint32_t neventcodes;
  uint32_t npulses;
  uint32_t noutputs;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_EVRCONFIGV3_H
