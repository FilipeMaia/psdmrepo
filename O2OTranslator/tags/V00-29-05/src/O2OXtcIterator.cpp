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

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/O2OXtcIterator.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/shared_ptr.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/O2OExceptions.h"
#include "O2OTranslator/O2OXtcScannerI.h"
#include "pdsdata/compress/CompressedXtc.hh"
#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/Xtc.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char logger[] = "O2OTranslator.O2OXtcIterator";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
O2OXtcIterator::O2OXtcIterator ( Xtc* xtc, O2OXtcScannerI* scanner, bool config )
  : Pds::XtcIterator( xtc )
  , m_scanner( scanner )
  , m_src()
  , m_config(config)
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
  if (xtc->contains.compressed()) {
    MsgLog(logger, debug, "Found compressed XTC, decompressing");
    boost::shared_ptr<Pds::Xtc> uxtc = Pds::CompressedXtc::uncompress(*xtc);
    return this->process(uxtc.get());
  }

  Pds::TypeId::Type type = xtc->contains.id() ;
  uint32_t version = xtc->contains.version() ;
  Pds::Level::Type level = xtc->src.level() ;

  // sanity check
  if ( xtc->sizeofPayload() < 0 ) {
    MsgLog(logger, error, "Negative payload size in XTC: " << xtc->sizeofPayload()
        << " level: " << int(level) << '#' << Pds::Level::name(level)
        << " type: " << int(type) << '#' << Pds::TypeId::name(type) << "/V" << version) ;
    throw O2OXTCGenException(ERR_LOC, "negative payload size") ;
  }

  m_src.push( xtc->src ) ;

  // say we are at the new XTC level
  if ( m_scanner ) m_scanner->levelStart ( xtc->src ) ;

  MsgLog(logger, debug, "O2OXtcIterator::process -- new xtc: "
              << int(type) << '#' << Pds::TypeId::name(type) << "/V" << version
              << " payload = " << xtc->sizeofPayload()
              << " damage: " << std::hex << std::showbase << xtc->damage.value()) ;

  int result = 1 ;
  if ( type == Pds::TypeId::Id_Xtc ) {

    // scan all sub-xtcs, this takes care of the bad damage bit (IncompleteContribution)
    this->iterate( xtc );

  } else if ( type == Pds::TypeId::Any ) {

    // NOTE: I do not know yet what this type is supposed to do, for now just ignore it

  } else if ( xtc->src.level() == Pds::Level::Source
      or xtc->src.level() == Pds::Level::Reporter
      or xtc->src.level() == Pds::Level::Event
      or xtc->src.level() == Pds::Level::Control ) {

    if (m_config) {
      m_scanner->configObject( xtc->payload(), xtc->sizeofPayload(), xtc->contains, m_src, xtc->damage ) ;
    } else {
      m_scanner->dataObject( xtc->payload(), xtc->sizeofPayload(), xtc->contains, m_src, xtc->damage ) ;
    }

  } else {

    MsgLog(logger, warning, "O2OXtcIterator::process -- data object at level "
                << int(level) << '#' << Pds::Level::name(level) << ": "
                << int(type) << '#' << Pds::TypeId::name(type) << "/V" << version);

  }

  // say we are done with this XTC level
  if ( m_scanner ) m_scanner->levelEnd ( xtc->src ) ;

  m_src.pop() ;

  return result ;
}

} // namespace O2OTranslator
