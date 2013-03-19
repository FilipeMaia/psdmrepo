#ifndef H5DATATYPES_EVRCONFIGV2_H
#define H5DATATYPES_EVRCONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrConfigV2.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "hdf5pp/Group.h"
#include "pdsdata/evr/ConfigV2.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace H5DataTypes {

//
// Helper type for Pds::EvrData::ConfigV2
//
class EvrConfigV2  {
public:

  typedef Pds::EvrData::ConfigV2 XtcType ;

  EvrConfigV2 () {}
  EvrConfigV2 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc.size() ; }

private:

  uint16_t beam;
  uint16_t rate;
  uint32_t npulses;
  uint32_t noutputs;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_EVRCONFIGV2_H
