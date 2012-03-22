#ifndef PYPDSDATA_LUSI_IPMFEXCONFIGV2_H
#define PYPDSDATA_LUSI_IPMFEXCONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Class IpmFexConfigV2.
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
#include "pdsdata/lusi/IpmFexConfigV2.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Lusi {

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

class IpmFexConfigV2 : public PdsDataType<IpmFexConfigV2,Pds::Lusi::IpmFexConfigV2> {
public:

  typedef PdsDataType<IpmFexConfigV2,Pds::Lusi::IpmFexConfigV2> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace Lusi
} // namespace pypdsdata

#endif // PYPDSDATA_LUSI_IPMFEXCONFIGV2_H
