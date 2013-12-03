#ifndef H5DATATYPES_BLDDATAPHASECAVITY_H
#define H5DATATYPES_BLDDATAPHASECAVITY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataPhaseCavity.
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
#include "hdf5pp/Type.h"
#include "pdsdata/psddl/bld.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::BldDataPhaseCavity
//
class BldDataPhaseCavity  {
public:

  typedef Pds::Bld::BldDataPhaseCavity XtcType ;

  BldDataPhaseCavity () {}
  BldDataPhaseCavity ( const XtcType& xtc ) ;

  ~BldDataPhaseCavity () {}

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;
  
  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

private:
  double fFitTime1;   /* in pico-seconds */
  double fFitTime2;   /* in pico-seconds */
  double fCharge1;    /* in pico-columbs */
  double fCharge2;    /* in pico-columbs */
};

} // namespace H5DataTypes

#endif // H5DATATYPES_BLDDATAPHASECAVITY_H
