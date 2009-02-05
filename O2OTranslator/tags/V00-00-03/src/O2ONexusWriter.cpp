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
#include <memory>
#include <cstdio>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/O2OExceptions.h"
#include "pdsdata/xtc/Level.hh"
#include "pdsdata/xtc/Sequence.hh"
#include "pdsdata/xtc/Src.hh"
#include "pdsdata/acqiris/ConfigV1.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace nexuspp ;

namespace {

  const char* logger = "NexusWriter" ;

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
  , m_file(0)
  , m_existingGroups()
{
  MsgLog( logger, debug, "O2ONexusWriter - open output file " << m_fileName ) ;
  m_file = nexuspp::NxppFile::open ( m_fileName.c_str(), nexuspp::NxppFile::CreateHdf5 );
  if ( not m_file ) {
    throw O2OFileOpenException(m_fileName) ;
  }
}

//--------------
// Destructor --
//--------------
O2ONexusWriter::~O2ONexusWriter ()
{
  MsgLog( logger, debug, "O2ONexusWriter - close output file " << m_fileName ) ;
  delete m_file ;
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
  if ( s < 0 or s >= int(sizeof buf) ) {
    MsgLog( logger, fatal, "snprintf conversion failed" ) ;
  }

  const char* topgroup = Pds::TransitionId::name(seq.service()) ;
  if ( m_existingGroups.count(topgroup) == 0 ) {
    MsgLog( logger, debug, "O2ONexusWriter::eventStart -- creating event group " << topgroup ) ;
    bool stat = m_file->createGroup ( topgroup, "NXentry" ) ;
    if ( not stat ) throw O2OTranslator::O2ONexusException( "NxppFile::createGroup" ) ;
    m_existingGroups.insert( topgroup ) ;
  } else {
    bool stat = m_file->openGroup ( topgroup, "NXentry" ) ;
    if ( not stat ) throw O2OTranslator::O2ONexusException( "NXopengroup" ) ;
  }
  MsgLog( logger, debug, "O2ONexusWriter::eventStart -- creating event group " << buf ) ;
  bool stat = m_file->createGroup ( buf, "NXentry" ) ;
  if ( not stat ) throw O2OTranslator::O2ONexusException( "NxppFile::createGroup" ) ;

}

void
O2ONexusWriter::eventEnd ( const Pds::Sequence& seq )
{
  MsgLog( logger, debug, "O2ONexusWriter::eventEnd " << Pds::TransitionId::name(seq.service()) ) ;
  MsgLog( logger, debug, "O2ONexusWriter::eventStart -- closing event group" ) ;

  bool stat = m_file->closeGroup() ;
  if ( not stat ) throw O2OTranslator::O2ONexusException( "NxppFile::closeGroup" ) ;
  stat = m_file->closeGroup() ;
  if ( not stat ) throw O2OTranslator::O2ONexusException( "NxppFile::closeGroup" ) ;
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


  std::auto_ptr<NxppDataSet<int> > ds ( m_file->makeDataSet<int>( "Acqiris::ConfigV1", 1 ) ) ;
  if ( not ds.get() ) throw O2OTranslator::O2ONexusException( "NxppFile::makeDataSet" ) ;

  if ( not ds->putData ( 0 ) ) throw O2OTranslator::O2ONexusException( "NxppDataSet::putData" ) ;

  bool stat =
    ds->addAttribute ( "sampInterval", data.sampInterval() ) and
    ds->addAttribute ( "delayTime", data.delayTime() ) and
    ds->addAttribute ( "nbrSamples", data.nbrSamples() ) and
    ds->addAttribute ( "nbrSegments", data.nbrSegments() ) and
    ds->addAttribute ( "coupling", data.coupling() ) and
    ds->addAttribute ( "bandwidth", data.bandwidth() ) and
    ds->addAttribute ( "fullScale", data.fullScale() ) and
    ds->addAttribute ( "offset", data.offset() ) and
    ds->addAttribute ( "trigCoupling", data.trigCoupling() ) and
    ds->addAttribute ( "trigInput", data.trigInput() ) and
    ds->addAttribute ( "trigSlope", data.trigSlope() ) and
    ds->addAttribute ( "trigLevel", data.trigLevel() ) ;
  if ( not stat ) throw O2OTranslator::O2ONexusException( "NxppDataSet::addAtribute" ) ;

}


} // namespace O2OTranslator
