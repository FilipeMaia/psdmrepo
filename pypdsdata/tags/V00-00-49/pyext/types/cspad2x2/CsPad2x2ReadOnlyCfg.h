#ifndef PYPDSDATA_CSPAD2X2_CSPAD2X2READONLYCFG_H
#define PYPDSDATA_CSPAD2X2_CSPAD2X2READONLYCFG_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2ReadOnlyCfg.
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
#include "pdsdata/cspad2x2/ConfigV1QuadReg.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

namespace pypdsdata {
namespace CsPad2x2 {

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

class CsPad2x2ReadOnlyCfg : public PdsDataType<CsPad2x2ReadOnlyCfg, Pds::CsPad2x2::CsPad2x2ReadOnlyCfg> {
public:

  typedef PdsDataType<CsPad2x2ReadOnlyCfg, Pds::CsPad2x2::CsPad2x2ReadOnlyCfg> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace CsPad2x2
} // namespace pypdsdata

#endif // PYPDSDATA_CSPAD2X2_CSPAD2X2READONLYCFG_H
