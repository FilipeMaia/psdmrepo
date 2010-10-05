#ifndef H5DATATYPES_LUSIIPMFEXCONFIGV1_H
#define H5DATATYPES_LUSIIPMFEXCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LusiIpmFexConfigV1.
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
#include "pdsdata/lusi/IpmFexConfigV1.hh"
#include "H5DataTypes/LusiDiodeFexConfigV1.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Lusi::IpmFexConfigV1
//
struct LusiIpmFexConfigV1_Data  {
  enum { NCHANNELS = Pds::Lusi::IpmFexConfigV1::NCHANNELS };
  LusiDiodeFexConfigV1 diode[NCHANNELS];
  float xscale;
  float yscale;
};

class LusiIpmFexConfigV1  {
public:

  typedef Pds::Lusi::IpmFexConfigV1 XtcType ;

  LusiIpmFexConfigV1 () {}
  LusiIpmFexConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  LusiIpmFexConfigV1_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_LUSIIPMFEXCONFIGV1_H
