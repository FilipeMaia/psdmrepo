#ifndef PYPDSDATA_EVRDATA_DATAV3_FIFOEVENT_H
#define PYPDSDATA_EVRDATA_DATAV3_FIFOEVENT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_DataV3_FIFOEvent.
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
#include "pdsdata/evr/DataV3.hh"

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

class DataV3_FIFOEvent : public PdsDataTypeEmbedded<DataV3_FIFOEvent,Pds::EvrData::DataV3::FIFOEvent> {
public:

  typedef PdsDataTypeEmbedded<DataV3_FIFOEvent,Pds::EvrData::DataV3::FIFOEvent> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace EvrData
} // namespace pypdsdata

#endif // PYPDSDATA_EVRDATA_DATAV3_FIFOEVENT_H
