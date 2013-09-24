#ifndef H5DATATYPES_IMPCONFIGV1_H
#define H5DATATYPES_IMPCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImpConfigV1.
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
#include "pdsdata/psddl/imp.ddl.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Imp::ConfigV1
//
class ImpConfigV1  {
public:

  typedef Pds::Imp::ConfigV1 XtcType ;

  ImpConfigV1 () {}
  ImpConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  uint32_t range; 
  uint32_t calRange; 
  uint32_t reset; 
  uint32_t biasData; 
  uint32_t calData; 
  uint32_t biasDacData; 
  uint32_t calStrobe; 
  uint32_t numberOfSamples; 
  uint32_t trigDelay; 
  uint32_t adcDelay; 

};

} // namespace H5DataTypes

#endif // H5DATATYPES_IMPCONFIGV1_H
