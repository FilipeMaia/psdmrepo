#ifndef PYPDSDATA_EVRDATA_DATAV3_H
#define PYPDSDATA_EVRDATA_DATAV3_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_DataV3.
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

class DataV3 : public PdsDataType<DataV3,Pds::EvrData::DataV3> {
public:

  typedef PdsDataType<DataV3,Pds::EvrData::DataV3> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace EvrData
} // namespace pypdsdata

#endif // PYPDSDATA_EVRDATA_DATAV3_H
