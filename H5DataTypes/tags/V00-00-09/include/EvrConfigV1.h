#ifndef H5DATATYPES_EVRCONFIGV1_H
#define H5DATATYPES_EVRCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrConfigV1.
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
#include "pdsdata/evr/ConfigV1.hh"

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
// Helper type for Pds::EvrData::ConfigV1
//
struct EvrConfigV1_Data {
  uint32_t npulses;
  uint32_t noutputs;
};

class EvrConfigV1  {
public:

  typedef Pds::EvrData::ConfigV1 XtcType ;

  EvrConfigV1 () {}
  EvrConfigV1 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

private:

  EvrConfigV1_Data m_data ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_EVRCONFIGV1_H
