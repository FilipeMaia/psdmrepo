#ifndef PYPDSDATA_TIMEPIX_DATAV2_H
#define PYPDSDATA_TIMEPIX_DATAV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Timepix_DataV2.
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
#include "pdsdata/timepix/DataV2.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Timepix {

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

class DataV2 : public PdsDataType<DataV2,Pds::Timepix::DataV2> {
public:

  typedef PdsDataType<DataV2,Pds::Timepix::DataV2> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // Very special "constructor" from XTC, it can handle both DataV1 container and DataV2.
  // For DataV1 it creates new DataV2 object which shuffles the data from DataV1.
  static DataV2* PyObject_FromXtc( const Pds::Xtc& xtc, PyObject* parent, destructor dtor=0 );

};

} // namespace Timepix
} // namespace pypdsdata

#endif // PYPDSDATA_TIMEPIX_DATAV2_H
