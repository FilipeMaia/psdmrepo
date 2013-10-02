#ifndef H5DATATYPES_BLDDATASPECTROMETERV0_H
#define H5DATATYPES_BLDDATASPECTROMETERV0_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataSpectrometerV0.
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
// Helper type for Pds::Bld::BldDataSpectrometerV0
//
class BldDataSpectrometerV0  {
public:

  typedef Pds::Bld::BldDataSpectrometerV0 XtcType ;

  BldDataSpectrometerV0 () {}
  BldDataSpectrometerV0 ( const XtcType& xtc ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

protected:

private:

  uint32_t  hproj[1024];
  uint32_t  vproj[256];

};

} // namespace H5DataTypes

#endif // H5DATATYPES_BLDDATASPECTROMETERV0_H
