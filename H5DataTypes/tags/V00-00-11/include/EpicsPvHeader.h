#ifndef H5DATATYPES_EPICSPVHEADER_H
#define H5DATATYPES_EPICSPVHEADER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpicsPvHeader.
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
#include "pdsdata/epics/EpicsPvData.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

struct EpicsPvHeader_Data  {
  int16_t pvId;
  int16_t dbrType;
  int16_t numElements;
};

class EpicsPvHeader  {
public:

  typedef Pds::EpicsPvHeader XtcType ;

  EpicsPvHeader () {}
  EpicsPvHeader ( const XtcType& xtc ) ;

  ~EpicsPvHeader () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  EpicsPvHeader_Data m_data ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_EPICSPVHEADER_H
