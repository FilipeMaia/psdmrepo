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
#include "pdsdata/bld/bldData.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

struct BldDataPhaseCavity_Data  {
  double fFitTime1;   /* in pico-seconds */ 
  double fFitTime2;   /* in pico-seconds */ 
  double fCharge1;    /* in pico-columbs */ 
  double fCharge2;    /* in pico-columbs */ 
};

class BldDataPhaseCavity  {
public:

  typedef Pds::BldDataPhaseCavity XtcType ;

  BldDataPhaseCavity () {}
  BldDataPhaseCavity ( const XtcType& xtc ) ;

  ~BldDataPhaseCavity () {}

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  BldDataPhaseCavity_Data m_data ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_BLDDATAPHASECAVITY_H
