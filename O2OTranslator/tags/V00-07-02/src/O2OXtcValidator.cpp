//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OXtcValidator...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/O2OXtcValidator.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iomanip>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/xtc/Xtc.hh"
#include "pdsdata/xtc/DetInfo.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
O2OXtcValidator::O2OXtcValidator ()
  : Pds::XtcIterator()
  , m_status(1)
{
}

//--------------
// Destructor --
//--------------
O2OXtcValidator::~O2OXtcValidator ()
{
}

// process one sub-XTC, returns >0 for success, 0 for error
int
O2OXtcValidator::process(Xtc* xtc)
{
  Pds::TypeId::Type type = xtc->contains.id() ;
  uint32_t version = xtc->contains.version() ;
  Pds::Level::Type level = xtc->src.level() ;
  
  if ( xtc->damage.value() != 0 ) {
    MsgLogRoot( error, "XTC damage: " << std::hex << xtc->damage.value() << std::dec
        << " level: " << int(level) << '#' << Pds::Level::name(level)
        << " type: " << int(type) << '#' << Pds::TypeId::name(type) << "/V" << version) ;
    m_status = 0 ;
    return m_status ;
  }
  
  if ( xtc->sizeofPayload() < 0 ) {
    MsgLogRoot( error, "Negative payload size in XTC: " << xtc->sizeofPayload()
        << " level: " << int(level) << '#' << Pds::Level::name(level)
        << " type: " << int(type) << '#' << Pds::TypeId::name(type) << "/V" << version) ;
    m_status = 0 ;
    return m_status ;
  }
  
  if ( type == Pds::TypeId::Id_Xtc ) {
    // scan all sub-xtcs
    this->iterate( xtc );
  }

  return m_status ;
}

} // namespace O2OTranslator
