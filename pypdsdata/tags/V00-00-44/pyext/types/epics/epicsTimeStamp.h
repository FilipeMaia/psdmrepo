#ifndef PYPDSDATA_EPICSTIMESTAMP_H
#define PYPDSDATA_EPICSTIMESTAMP_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class epicsTimeStamp.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "../PdsDataTypeEmbedded.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/epics/EpicsDbrTools.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Epics {

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

class epicsTimeStamp : public PdsDataTypeEmbedded<epicsTimeStamp,Pds::Epics::epicsTimeStamp> {
public:

  typedef PdsDataTypeEmbedded<epicsTimeStamp,Pds::Epics::epicsTimeStamp> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace Epics
} // namespace pypdsdata

#endif // PYPDSDATA_EPICSTIMESTAMP_H
