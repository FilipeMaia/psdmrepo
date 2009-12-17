#ifndef H5DATATYPES_PULNIXTM6740CONFIGV1_H
#define H5DATATYPES_PULNIXTM6740CONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PulnixTM6740ConfigV1.
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
#include "pdsdata/pulnix/TM6740ConfigV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

struct PulnixTM6740ConfigV1_Data {
  uint16_t vref;
  uint16_t gain_a;
  uint16_t gain_b;
  uint8_t gain_balance;
  uint16_t shutter_width;
  uint8_t output_resolution;
  uint8_t output_resolution_bits;
  uint8_t horizontal_binning;
  uint8_t vertical_binning;
  uint8_t lookuptable_mode;
};

/**
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

class PulnixTM6740ConfigV1  {
public:

  typedef Pds::Pulnix::TM6740ConfigV1 XtcType ;

  PulnixTM6740ConfigV1 () {}
  PulnixTM6740ConfigV1 ( const Pds::Pulnix::TM6740ConfigV1& config ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static void store ( const Pds::Pulnix::TM6740ConfigV1& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

private:
  PulnixTM6740ConfigV1_Data m_data ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_PULNIXTM6740CONFIGV1_H
