#ifndef PYPDSDATA_XTC_H
#define PYPDSDATA_XTC_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Xtc.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "types/PdsDataType.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/xtc/Xtc.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {

/**
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class Xtc : public PdsDataType<Xtc,Pds::Xtc> {
public:

  typedef PdsDataType<Xtc,Pds::Xtc> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // Check object type
  static bool Xtc_Check( PyObject* obj );

  // REturns a pointer to Pds object
  static Pds::Xtc* Xtc_AsPds( PyObject* obj );
};

} // namespace pypdsdata

#endif // PYPDSDATA_XTC_H
