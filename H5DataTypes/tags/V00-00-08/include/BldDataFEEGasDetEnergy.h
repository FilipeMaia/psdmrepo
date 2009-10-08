#ifndef H5DATATYPES_BLDDATAFEEGASDETENERGY_H
#define H5DATATYPES_BLDDATAFEEGASDETENERGY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataFEEGasDetEnergy.
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


struct BldDataFEEGasDetEnergy_Data  {
  double f_11_ENRC;   /* in mJ */
  double f_12_ENRC;   /* in mJ */
  double f_21_ENRC;   /* in mJ */
  double f_22_ENRC;   /* in mJ */
};

class BldDataFEEGasDetEnergy  {
public:

  typedef Pds::BldDataFEEGasDetEnergy XtcType ;

  BldDataFEEGasDetEnergy () {}
  BldDataFEEGasDetEnergy ( const XtcType& xtc ) ;

  ~BldDataFEEGasDetEnergy () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  BldDataFEEGasDetEnergy_Data m_data ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_BLDDATAFEEGASDETENERGY_H
