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
#include "pdsdata/psddl/bld.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::BldDataFEEGasDetEnergy
//
class BldDataFEEGasDetEnergy  {
public:

  typedef Pds::Bld::BldDataFEEGasDetEnergy XtcType ;

  BldDataFEEGasDetEnergy () {}
  BldDataFEEGasDetEnergy ( const XtcType& xtc ) ;

  ~BldDataFEEGasDetEnergy () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

private:
  double f_11_ENRC;   /* in mJ */
  double f_12_ENRC;   /* in mJ */
  double f_21_ENRC;   /* in mJ */
  double f_22_ENRC;   /* in mJ */
};

} // namespace H5DataTypes

#endif // H5DATATYPES_BLDDATAFEEGASDETENERGY_H
