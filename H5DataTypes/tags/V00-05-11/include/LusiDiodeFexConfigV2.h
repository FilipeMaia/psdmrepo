#ifndef H5DATATYPES_LUSIDIODEFEXCONFIGV2_H
#define H5DATATYPES_LUSIDIODEFEXCONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LusiDiodeFexConfigV2.
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
#include "pdsdata/psddl/lusi.ddl.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Lusi::DiodeFexConfigV2
//
class LusiDiodeFexConfigV2  {
public:

  typedef Pds::Lusi::DiodeFexConfigV2 XtcType ;

  LusiDiodeFexConfigV2 () {}
  LusiDiodeFexConfigV2 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  enum { NRANGES = Pds::Lusi::DiodeFexConfigV2::NRANGES };
  float base [NRANGES];
  float scale[NRANGES];

};

} // namespace H5DataTypes

#endif // H5DATATYPES_LUSIDIODEFEXCONFIGV2_H
