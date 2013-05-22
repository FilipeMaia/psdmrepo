#ifndef H5DATATYPES_ACQIRISTDCDATAV1_H
#define H5DATATYPES_ACQIRISTDCDATAV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisTdcDataV1.
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
#include "pdsdata/acqiris/TdcDataV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
//  Helper class for Pds::Acqiris::TdcDataV1 class
//

class AcqirisTdcDataV1  {
public:

  typedef Pds::Acqiris::TdcDataV1 XtcType ;

  AcqirisTdcDataV1 () ;
  AcqirisTdcDataV1 ( size_t size, const XtcType* xtcData ) ;

  ~AcqirisTdcDataV1 () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

protected:
private:

#pragma pack(push,2)
  struct AcqirisTdcDataV1_Data  {
    uint8_t source;
    uint8_t overflow;
    uint32_t value;
  };
#pragma pack(pop)

  size_t m_size;
  AcqirisTdcDataV1_Data* m_data;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_ACQIRISTDCDATAV1_H
