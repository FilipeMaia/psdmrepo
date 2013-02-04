#ifndef H5DATATYPES_LUSIIPMFEXCONFIGV2_H
#define H5DATATYPES_LUSIIPMFEXCONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LusiIpmFexConfigV2.
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
#include "pdsdata/lusi/IpmFexConfigV2.hh"
#include "H5DataTypes/LusiDiodeFexConfigV2.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Lusi::IpmFexConfigV2
//
struct LusiIpmFexConfigV2_Data  {
  enum { NCHANNELS = Pds::Lusi::IpmFexConfigV2::NCHANNELS };
  LusiDiodeFexConfigV2 diode[NCHANNELS];
  float xscale;
  float yscale;
};

class LusiIpmFexConfigV2  {
public:

  typedef Pds::Lusi::IpmFexConfigV2 XtcType ;

  LusiIpmFexConfigV2 () {}
  LusiIpmFexConfigV2 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  LusiIpmFexConfigV2_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_LUSIIPMFEXCONFIGV2_H
