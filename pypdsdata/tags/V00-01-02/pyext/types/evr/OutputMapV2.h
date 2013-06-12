#ifndef PYPDSDATA_EVRDATA_OUTPUTMAPV2_H
#define PYPDSDATA_EVRDATA_OUTPUTMAPV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OutputMapV2.
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
#include "pdsdata/evr/OutputMapV2.hh"

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

class OutputMapV2 : public PdsDataTypeEmbedded<OutputMapV2,Pds::EvrData::OutputMapV2> {
public:

  typedef PdsDataTypeEmbedded<OutputMapV2,Pds::EvrData::OutputMapV2> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );
  
  /// access to Conn enum type
  static pypdsdata::EnumType& connEnum() ;

};

} // namespace EvrData
} // namespace pypdsdata

#endif // PYPDSDATA_EVRDATA_OUTPUTMAPV2_H
