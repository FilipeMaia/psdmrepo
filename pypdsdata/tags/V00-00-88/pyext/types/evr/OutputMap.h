#ifndef PYPDSDATA_EVRDATA_OUTPUTMAP_H
#define PYPDSDATA_EVRDATA_OUTPUTMAP_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OutputMap.
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
#include "pdsdata/evr/OutputMap.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {

class EnumType;

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

class OutputMap : public PdsDataTypeEmbedded<OutputMap,Pds::EvrData::OutputMap> {
public:

  typedef PdsDataTypeEmbedded<OutputMap,Pds::EvrData::OutputMap> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );
  
  /// access to Conn enum type
  static pypdsdata::EnumType& connEnum() ;

};

} // namespace EvrData
} // namespace pypdsdata

#endif // PYPDSDATA_EVRDATA_OUTPUTMAP_H
