#ifndef PYPDSDATA_CSPAD2X2_CSPAD2X2GAINMAPCFG_H
#define PYPDSDATA_CSPAD2X2_CSPAD2X2GAINMAPCFG_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2GainMapCfg.
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
#include "pdsdata/psddl/cspad2x2.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

namespace pypdsdata {
namespace CsPad2x2 {

/// @addtogroup pypdsdata

/**
 *  @ingroup pypdsdata
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class CsPad2x2GainMapCfg : public PdsDataType<CsPad2x2GainMapCfg, Pds::CsPad2x2::CsPad2x2GainMapCfg> {
public:

  typedef PdsDataType<CsPad2x2GainMapCfg, Pds::CsPad2x2::CsPad2x2GainMapCfg> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace CsPad2x2
} // namespace pypdsdata

#endif // PYPDSDATA_CSPAD2X2_CSPAD2X2GAINMAPCFG_H
