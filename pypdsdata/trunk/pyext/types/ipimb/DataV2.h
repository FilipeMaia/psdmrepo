#ifndef PYPDSDATA_IPIMB_DATAV2_H
#define PYPDSDATA_IPIMB_DATAV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataV2.
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
#include "pdsdata/ipimb/DataV2.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {
namespace Ipimb {

/**
 *  This software was developed for the LCLS project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class DataV2 : public PdsDataType<DataV2,Pds::Ipimb::DataV2> {
public:

  typedef PdsDataType<DataV2,Pds::Ipimb::DataV2> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace Ipimb
} // namespace pypdsdata

#endif // PYPDSDATA_IPIMB_DATAV2_H
