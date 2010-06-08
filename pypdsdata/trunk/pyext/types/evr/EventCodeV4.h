#ifndef PYPDSDATA_EVRDATA_EVENTCODEV4_H
#define PYPDSDATA_EVRDATA_EVENTCODEV4_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_EventCodeV4.
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
#include "pdsdata/evr/EventCodeV4.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace EvrData {

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

class EventCodeV4 : public PdsDataTypeEmbedded<EventCodeV4,Pds::EvrData::EventCodeV4> {
public:

  typedef PdsDataTypeEmbedded<EventCodeV4,Pds::EvrData::EventCodeV4> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace EvrData
} // namespace pypdsdata

#endif // PYPDSDATA_EVRDATA_EVENTCODEV4_H
