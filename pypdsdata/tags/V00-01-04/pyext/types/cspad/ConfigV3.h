#ifndef PYPDSDATA_CSPAD_CONFIGV3_H
#define PYPDSDATA_CSPAD_CONFIGV3_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV3.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "../PdsDataType.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/cspad/ConfigV3.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {
namespace CsPad {

/**
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class ConfigV3 : public PdsDataType<ConfigV3, Pds::CsPad::ConfigV3> {
public:

  typedef PdsDataType<ConfigV3, Pds::CsPad::ConfigV3> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& out) const;

};

} // namespace CsPad
} // namespace pypdsdata

#endif // PYPDSDATA_CSPAD_CONFIGV3_H
