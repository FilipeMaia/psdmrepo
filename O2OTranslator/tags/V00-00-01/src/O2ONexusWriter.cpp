//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2ONexusWriter...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/O2ONexusWriter.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <cstdio>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/O2OExceptions.h"
#include "pdsdata/xtc/Level.hh"
#include "pdsdata/xtc/Sequence.hh"
#include "pdsdata/xtc/Src.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "NexusWriter" ;

  void makeGroup ( NXhandle fileId, const char* group, const char* type )
  {
    int status = NXmakegroup( fileId, group, type );
    if ( status != NX_OK ) throw O2OTranslator::O2ONexusException( "NXmakegroup" ) ;

    status = NXopengroup( fileId, group, type );
    if ( status != NX_OK ) throw O2OTranslator::O2ONexusException( "NXopengroup" ) ;
  }

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
O2ONexusWriter::O2ONexusWriter ( const std::string& fileName )
  : O2OXtcScannerI()
  , m_fileName( fileName )
  , m_fileId()
  , m_existingGroups()
{
  MsgLog( logger, debug, "O2ONexusWriter - open output file " << m_fileName ) ;
  int status = NXopen ( m_fileName.c_str(), NXACC_CREATE5, &m_fileId );
  if ( status != NX_OK ) {
    throw O2OFileOpenException(m_fileName) ;
  }
}

//--------------
// Destructor --
//--------------
O2ONexusWriter::~O2ONexusWriter ()
{
  MsgLog( logger, debug, "O2ONexusWriter - close output file " << m_fileName ) ;
  int status = NXclose( &m_fileId ) ;
  if ( status != NX_OK ) throw O2OTranslator::O2ONexusException( "NXclose" ) ;
}

// signal start/end of the event
void
O2ONexusWriter::eventStart ( const Pds::Sequence& seq )
{
  MsgLog( logger, debug, "O2ONexusWriter::eventStart " << Pds::TransitionId::name(seq.service())
          << " seq.type=" << seq.type()
          << " seq.service=" << Pds::TransitionId::name(seq.service()) ) ;

  // for every event we create new group in a file, group name should include event time
  char buf[32] ;
  int s = snprintf ( buf, sizeof buf, "%08X:%08X", seq.high(), seq.low() ) ;
  if ( s < 0 or s >= sizeof buf ) {
    MsgLog( logger, fatal, "snprintf conversion failed" ) ;
  }

  const char* topgroup = Pds::TransitionId::name(seq.service()) ;
  if ( m_existingGroups.count(topgroup) == 0 ) {
    MsgLog( logger, debug, "O2ONexusWriter::eventStart -- creating event group " << topgroup ) ;
    ::makeGroup ( m_fileId, topgroup, "NXentry" ) ;
    m_existingGroups.insert( topgroup ) ;
  } else {
    int status = NXopengroup( m_fileId, topgroup, "NXentry" );
    if ( status != NX_OK ) throw O2OTranslator::O2ONexusException( "NXopengroup" ) ;
  }
  MsgLog( logger, debug, "O2ONexusWriter::eventStart -- creating event group " << buf ) ;
  ::makeGroup ( m_fileId, buf, "NXentry" ) ;

}

void
O2ONexusWriter::eventEnd ( const Pds::Sequence& seq )
{
  MsgLog( logger, debug, "O2ONexusWriter::eventEnd " << Pds::TransitionId::name(seq.service()) ) ;
  MsgLog( logger, debug, "O2ONexusWriter::eventStart -- closing event group" ) ;

  int status = NXclosegroup( m_fileId );
  if ( status != NX_OK ) throw O2ONexusException( "NXclosegroup" ) ;
  status = NXclosegroup( m_fileId );
  if ( status != NX_OK ) throw O2ONexusException( "NXclosegroup" ) ;
}

// signal start/end of the level
void
O2ONexusWriter::levelStart ( const Pds::Src& src )
{
  MsgLog( logger, debug, "O2ONexusWriter::levelStart " << Pds::Level::name(src.level()) ) ;
}

void
O2ONexusWriter::levelEnd ( const Pds::Src& src )
{
  MsgLog( logger, debug, "O2ONexusWriter::levelEnd " << Pds::Level::name(src.level()) ) ;
}

// visit the data object
void
O2ONexusWriter::dataObject ( const Pds::WaveformV1& data, const Pds::Src& src )
{
  MsgLog( logger, debug, "O2ONexusWriter::dataObject WaveformV1 " << Pds::Level::name(src.level()) ) ;
}

void
O2ONexusWriter::dataObject ( const Pds::Acqiris::ConfigV1& data, const Pds::Src& src )
{
  MsgLog( logger, debug, "O2ONexusWriter::dataObject Acqiris::ConfigV1 " << Pds::Level::name(src.level()) ) ;
}


} // namespace O2OTranslator
