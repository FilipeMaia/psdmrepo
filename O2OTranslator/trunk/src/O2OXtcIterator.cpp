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
#include "Lusi/Lusi.h"

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

  // sanity check
  if ( xtc->sizeofPayload() < 0 ) {
    MsgLogRoot( error, "Negative payload size in XTC: " << xtc->sizeofPayload()
        << " level: " << Pds::Level::name(xtc->src.level())
        << " type: " << Pds::TypeId::name(type) << "/" << version) ;
    throw O2OXTCGenException("negative payload size") ;
  }

  m_src.push( xtc->src ) ;

  // say we are at the new XTC level
  if ( m_scanner ) m_scanner->levelStart ( xtc->src ) ;

  MsgLogRoot( debug, "O2OXtcIterator::process -- new xtc: "
              << Pds::TypeId::name(type) << "/" << version
              << " payload = " << xtc->sizeofPayload() ) ;

  int result = 1 ;
  if ( type == Pds::TypeId::Id_Xtc ) {

    // scan all sub-xtcs
    this->iterate( xtc );

  } else if ( type == Pds::TypeId::Any ) {

    // NOTE: I do not know yet what this type is supposed to do, for now just ignore it

  } else if ( xtc->src.level() == Pds::Level::Source
      or xtc->src.level() == Pds::Level::Reporter
      or xtc->src.level() == Pds::Level::Control ) {

    m_scanner->dataObject( xtc->payload(), xtc->contains, m_src ) ;

  } else {

    MsgLogRoot( warning, "O2OXtcIterator::process -- data object at "
                << Pds::Level::name(xtc->src.level()) << " level: "
                << Pds::TypeId::name(type) << "/" << version ) ;

  }

  // say we are done with this XTC level
  if ( m_scanner ) m_scanner->levelEnd ( xtc->src ) ;

  m_src.pop() ;

  return result ;
}

} // namespace O2OTranslator
