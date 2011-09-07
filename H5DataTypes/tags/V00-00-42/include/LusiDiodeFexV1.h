#ifndef H5DATATYPES_LUSIDIODEFEXV1_H
#define H5DATATYPES_LUSIDIODEFEXV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LusiDiodeFexV1.
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
#include "pdsdata/lusi/DiodeFexV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Lusi::DiodeFexV1
//
struct LusiDiodeFexV1_Data  {
  float value;
};

class LusiDiodeFexV1  {
public:

  typedef Pds::Lusi::DiodeFexV1 XtcType ;

  LusiDiodeFexV1 () {}
  LusiDiodeFexV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  LusiDiodeFexV1_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_LUSIDIODEFEXV1_H
