#ifndef PYPDSDATA_EVRDATA_IOCHANNEL_H
#define PYPDSDATA_EVRDATA_IOCHANNEL_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_IOChannel.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/evr/IOChannel.hh"

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

class IOChannel : public PdsDataType<IOChannel,Pds::EvrData::IOChannel> {
public:

  typedef PdsDataType<IOChannel,Pds::EvrData::IOChannel> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace EvrData
} // namespace pypdsdata

#endif // PYPDSDATA_EVRDATA_IOCHANNEL_H
