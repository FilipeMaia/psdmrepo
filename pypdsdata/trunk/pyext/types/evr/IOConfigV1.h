#ifndef PYPDSDATA_EVRDATA_IOCONFIGV1_H
#define PYPDSDATA_EVRDATA_IOCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_IOConfigV1.
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
#include "pdsdata/evr/IOConfigV1.hh"

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

class IOConfigV1 : public PdsDataType<IOConfigV1,Pds::EvrData::IOConfigV1> {
public:

  typedef PdsDataType<IOConfigV1,Pds::EvrData::IOConfigV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace EvrData
} // namespace pypdsdata

#endif // PYPDSDATA_EVRDATA_IOCONFIGV1_H
