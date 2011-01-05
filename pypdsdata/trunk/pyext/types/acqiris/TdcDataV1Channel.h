#ifndef PYPDSDATA_ACQIRIS_TDCDATAV1CHANNEL_H
#define PYPDSDATA_ACQIRIS_TDCDATAV1CHANNEL_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TdcDataV1Channel.
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
#include "pdsdata/acqiris/TdcDataV1.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Acqiris {

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

class TdcDataV1Channel : public PdsDataType<TdcDataV1Channel,class Pds::Acqiris::TdcDataV1::Channel> {
public:

  typedef PdsDataType<TdcDataV1Channel,class Pds::Acqiris::TdcDataV1::Channel> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& out) const;
};

} // namespace Acqiris
} // namespace pypdsdata

#endif // PYPDSDATA_ACQIRIS_TDCDATAV1CHANNEL_H
