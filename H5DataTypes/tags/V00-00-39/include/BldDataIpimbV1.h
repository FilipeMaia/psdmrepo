#ifndef H5DATATYPES_BLDDATAIPIMBV1_H
#define H5DATATYPES_BLDDATAIPIMBV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataIpimbV1.
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
#include "pdsdata/bld/bldData.hh"
#include "H5DataTypes/IpimbConfigV2.h"
#include "H5DataTypes/IpimbDataV2.h"
#include "H5DataTypes/LusiIpmFexV1.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

struct BldDataIpimbV1_Data {
  IpimbDataV2_Data    ipimbData;
  IpimbConfigV2_Data  ipimbConfig;
  LusiIpmFexV1_Data   ipmFexData;
};

/**
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class BldDataIpimbV1  {
public:

  typedef Pds::BldDataIpimbV1 XtcType ;

  BldDataIpimbV1 () {}
  BldDataIpimbV1 ( const XtcType& xtc ) ;

  ~BldDataIpimbV1 () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

private:
  BldDataIpimbV1_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_BLDDATAIPIMBV1_H
