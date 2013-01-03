#ifndef H5DATATYPES_LUSIIPMFEXV1_H
#define H5DATATYPES_LUSIIPMFEXV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LusiIpmFexV1.
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
#include "pdsdata/lusi/IpmFexV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Lusi::IpmFexV1
//
struct LusiIpmFexV1_Data  {
  enum { CHSIZE = 4 };
  float channel[CHSIZE];
  float sum;
  float xpos;
  float ypos;
};

class LusiIpmFexV1  {
public:

  typedef Pds::Lusi::IpmFexV1 XtcType ;

  LusiIpmFexV1 () {}
  LusiIpmFexV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  LusiIpmFexV1_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_LUSIIPMFEXV1_H
