#ifndef PYPDSDATA_PRINCETON_CONFIGV4_H
#define PYPDSDATA_PRINCETON_CONFIGV4_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV4.
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
#include "pdsdata/princeton/ConfigV4.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {
namespace Princeton {

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

class ConfigV4 : public PdsDataType<ConfigV4,Pds::Princeton::ConfigV4> {
public:

  typedef PdsDataType<ConfigV4,Pds::Princeton::ConfigV4> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // Dump object info to a stream
  void print(std::ostream& out) const;

};

} // namespace Princeton
} // namespace pypdsdata

#endif // PYPDSDATA_PRINCETON_CONFIGV4_H
