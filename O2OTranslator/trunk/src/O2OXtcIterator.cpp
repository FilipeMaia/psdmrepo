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

  MsgLogRoot( debug, "O2OXtcIterator::process -- new xtc: "
              << Pds::TypeId::name(type) << "/" << version ) ;

  int result = 1 ;
  if ( type == Pds::TypeId::Id_Xtc ) {

    // say we are at the new XTC level
    if ( m_scanner ) m_scanner->levelStart ( xtc->src ) ;

    // scan all sub-xtcs
    O2OXtcIterator iter( xtc, m_scanner );
    iter.iterate();

    // say we are done with this XTC level
    if ( m_scanner ) m_scanner->levelEnd ( xtc->src ) ;

  } else if ( type == Pds::TypeId::Any ) {

    // NOTE: I do not know yet what this type is supposed to do, for now just ignore it

  } else if ( xtc->src.level() == Pds::Level::Source ) {

    const Pds::DetInfo& detInfo = static_cast<const Pds::DetInfo&>(xtc->src);
    m_scanner->dataObject( xtc->payload(), xtc->contains, detInfo ) ;

  } else {

    MsgLogRoot( error, "O2OXtcIterator::process -- data object not at Source level: "
                << Pds::TypeId::name(type) << "/" << version ) ;
    result = 0;
  }

  return result ;
}

} // namespace O2OTranslator
