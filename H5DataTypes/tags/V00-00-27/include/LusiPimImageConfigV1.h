#ifndef H5DATATYPES_LUSIPIMIMAGECONFIGV1_H
#define H5DATATYPES_LUSIPIMIMAGECONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LusiPimImageConfigV1.
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
#include "pdsdata/lusi/PimImageConfigV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Lusi::PimImageConfigV1
//
struct LusiPimImageConfigV1_Data  {
  float xscale;
  float yscale;
};

class LusiPimImageConfigV1  {
public:

  typedef Pds::Lusi::PimImageConfigV1 XtcType ;

  LusiPimImageConfigV1 () {}
  LusiPimImageConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  LusiPimImageConfigV1_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_LUSIPIMIMAGECONFIGV1_H
