#ifndef H5DATATYPES_BLDDATAIPIMB_H
#define H5DATATYPES_BLDDATAIPIMB_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataIpimb.
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
#include "H5DataTypes/IpimbConfigV1.h"
#include "H5DataTypes/IpimbDataV1.h"
#include "H5DataTypes/LusiIpmFexV1.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

struct BldDataIpimb_Data {
  IpimbDataV1_Data    ipimbData;
  IpimbConfigV1_Data  ipimbConfig;
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

class BldDataIpimb  {
public:

  typedef Pds::BldDataIpimb XtcType ;

  BldDataIpimb () {}
  BldDataIpimb ( const XtcType& xtc ) ;

  ~BldDataIpimb () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

private:
  BldDataIpimb_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_BLDDATAIPIMB_H
