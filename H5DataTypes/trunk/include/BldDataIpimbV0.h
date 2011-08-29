#ifndef H5DATATYPES_BLDDATAIPIMBV0_H
#define H5DATATYPES_BLDDATAIPIMBV0_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataIpimbV0.
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

class BldDataIpimbV0  {
public:

  typedef Pds::BldDataIpimbV0 XtcType ;

  BldDataIpimbV0 () {}
  BldDataIpimbV0 ( const XtcType& xtc ) ;

  ~BldDataIpimbV0 () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

private:

  IpimbDataV1    ipimbData;
  IpimbConfigV1  ipimbConfig;
  LusiIpmFexV1   ipmFexData;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_BLDDATAIPIMBV0_H
