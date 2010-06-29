//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OXtcIterator...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/O2OXtcIterator.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/O2OExceptions.h"
#include "O2OTranslator/O2OXtcScannerI.h"
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
O2OXtcIterator::O2OXtcIterator ( Xtc* xtc, O2OXtcScannerI* scanner )
  : Pds::XtcIterator( xtc )
  , m_scanner( scanner )
  , m_src()
{
}

//--------------
// Destructor --
//--------------
O2OXtcIterator::~O2OXtcIterator ()
{
}

// process one sub-XTC, returns >0 for success, 0 for error
int
O2OXtcIterator::process(Xtc* xtc)
{
  Pds::TypeId::Type type = xtc->contains.id() ;
  uint32_t version = xtc->contains.version() ;
  Pds::Level::Type level = xtc->src.level() ;

  // sanity check
  if ( xtc->sizeofPayload() < 0 ) {
    MsgLogRoot( error, "Negative payload size in XTC: " << xtc->sizeofPayload()
        << " level: " << int(level) << '#' << Pds::Level::name(level)
        << " type: " << int(type) << '#' << Pds::TypeId::name(type) << "/V" << version) ;
    throw O2OXTCGenException("negative payload size") ;
  }

  m_src.push( xtc->src ) ;

  // say we are at the new XTC level
  if ( m_scanner ) m_scanner->levelStart ( xtc->src ) ;

  MsgLogRoot( debug, "O2OXtcIterator::process -- new xtc: "
              << int(type) << '#' << Pds::TypeId::name(type) << "/V" << version
              << " payload = " << xtc->sizeofPayload()
              << " damage: " << std::hex << std::showbase << xtc->damage.value() ) ;

  int result = 1 ;
  if ( type == Pds::TypeId::Id_Xtc ) {

    if ( xtc->damage.value() & Pds::Damage::DroppedContribution ) {
      //some types of damage cause troubles, filter them
      // skip damaged data
      MsgLogRoot( warning, "O2OXtcIterator::process -- damaged container xtc: "
                  << int(type) << '#' << Pds::TypeId::name(type) << "/V" << version
                  << " payload = " << xtc->sizeofPayload()
                  << " damage: " << std::hex << std::showbase << xtc->damage.value() ) ;
    } else {
      // scan all sub-xtcs
      this->iterate( xtc );
    }

  } else if ( type == Pds::TypeId::Any ) {

    // NOTE: I do not know yet what this type is supposed to do, for now just ignore it

  } else if ( xtc->src.level() == Pds::Level::Source
      or xtc->src.level() == Pds::Level::Reporter
      or xtc->src.level() == Pds::Level::Control ) {

    if ( xtc->damage.value() ) {
      // skip damaged data
      MsgLogRoot( warning, "O2OXtcIterator::process -- damaged data xtc: "
                  << int(type) << '#' << Pds::TypeId::name(type) << "/V" << version
                  << " payload = " << xtc->sizeofPayload()
                  << " damage: " << std::hex << std::showbase << xtc->damage.value() ) ;
    } else {
      m_scanner->dataObject( xtc->payload(), xtc->sizeofPayload(), xtc->contains, m_src ) ;
    }

  } else {

    MsgLogRoot( warning, "O2OXtcIterator::process -- data object at level "
                << int(level) << '#' << Pds::Level::name(level) << ": "
                << int(type) << '#' << Pds::TypeId::name(type) << "/V" << version ) ;

  }

  // say we are done with this XTC level
  if ( m_scanner ) m_scanner->levelEnd ( xtc->src ) ;

  m_src.pop() ;

  return result ;
}

} // namespace O2OTranslator
