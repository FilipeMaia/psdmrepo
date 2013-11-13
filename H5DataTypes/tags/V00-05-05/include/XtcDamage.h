#ifndef H5DATATYPES_XTCDAMAGE_H
#define H5DATATYPES_XTCDAMAGE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcDamage.
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
#include "pdsdata/xtc/Damage.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

/// @addtogroup H5DataTypes

/**
 *  @ingroup H5DataTypes
 *
 *  @brief Persistent data type for Xtc Damage
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class XtcDamage  {
public:

  XtcDamage () {}
  XtcDamage(Pds::Damage damage);

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

protected:

private:

  uint32_t bits;
  uint8_t DroppedContribution;
  uint8_t OutOfOrder;
  uint8_t OutOfSynch;
  uint8_t UserDefined;
  uint8_t IncompleteContribution;
  uint8_t userBits;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_XTCDAMAGE_H
