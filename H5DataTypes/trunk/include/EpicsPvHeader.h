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
#include "pdsdata/psddl/epics.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

class EpicsPvHeader  {
public:

  typedef Pds::Epics::EpicsPvHeader XtcType ;

  EpicsPvHeader () {}
  EpicsPvHeader ( const XtcType& xtc ) ;

  ~EpicsPvHeader () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  int16_t pvId;
  int16_t dbrType;
  int16_t numElements;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_EPICSPVHEADER_H
