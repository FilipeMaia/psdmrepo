#ifndef H5DATATYPES_EVRCONFIGV1_H
#define H5DATATYPES_EVRCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrConfigV1.
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
#include "pdsdata/psddl/evr.ddl.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::EvrData::ConfigV1
//
class EvrConfigV1  {
public:

  typedef Pds::EvrData::ConfigV1 XtcType ;

  EvrConfigV1 () {}
  EvrConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc._sizeof() ; }

private:

  uint32_t npulses;
  uint32_t noutputs;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_EVRCONFIGV1_H
