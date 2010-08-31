//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OXtcMerger...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/O2OXtcMerger.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <map>
#include <iomanip>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/O2OExceptions.h"
#include "pdsdata/xtc/TransitionId.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "O2OXtcMerger" ;


  Pds::Dgram* dg_copy ( Pds::Dgram*dg )
  {
    // make a copy
    char* dgbuf = (char*)dg ;
    size_t dgsize = sizeof(Pds::Dgram) + dg->xtc.sizeofPayload();
    char* buf = new char[dgsize] ;
    std::copy( dgbuf, dgbuf+dgsize, buf ) ;
    return (Pds::Dgram*)buf ;
  }

  // read next datagram from a stream, skip some problematic stuff
  Pds::Dgram* read_dg (O2OTranslator::O2OXtcDechunk* stream)
  {
    while ( true ) {
      Pds::Dgram* dg = stream->next();
      // skip Disable transitions, they mess up datagram order sometimes
      if ( not dg or dg->seq.service() != Pds::TransitionId::Disable ) {
        return dg;
      }
    }
  }
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
O2OXtcMerger::O2OXtcMerger ( const std::list<O2OXtcFileName>& files,
                             size_t maxDgSize,
                             MergeMode mode,
                             bool skipDamaged,
                             double l1OffsetSec )
  : m_streams()
  , m_dgrams()
  , m_mode(mode)
  , m_l1OffsetSec(int(l1OffsetSec))
  , m_l1OffsetNsec(int((l1OffsetSec-m_l1OffsetSec)*1e9))
{
  // check that we have at least one input stream
  if ( files.empty() ) {
    throw O2OTranslator::O2OArgumentException( "O2OXtcMerger: file list is empty" ) ;
  }

  typedef std::map< unsigned, std::list<O2OXtcFileName> > StreamMap ;
  StreamMap streamMap ;

  // separate files from different streams
  unsigned stream = 0 ;
  for ( std::list<O2OXtcFileName>::const_iterator it = files.begin() ; it != files.end() ; ++ it ) {
    if ( mode == FileName ) {
      stream = it->stream() ;
    } else if ( mode == NoChunking ) {
      stream ++ ;
    }
    streamMap[stream].push_back( *it );
    MsgLog( logger, trace, "O2OXtcMerger -- file: " << it->path() << " stream: " << stream ) ;
  }

  // create all streams
  m_streams.reserve( streamMap.size() ) ;
  m_dgrams.reserve( streamMap.size() ) ;
  for ( StreamMap::const_iterator it = streamMap.begin() ; it != streamMap.end() ; ++ it ) {

    std::list<O2OXtcFileName> streamFiles = it->second ;

    WithMsgLog( logger, info, out ) {
      out << "O2OXtcMerger -- stream: " << it->first ;
      for ( std::list<O2OXtcFileName>::const_iterator it = streamFiles.begin() ; it != streamFiles.end() ; ++ it ) {
        out << "\n             " << it->path() ;
      }
    }

    if ( mode == FileName ) {
      // order according to chunk number
      streamFiles.sort() ;
    }
    // create new stream
    O2OXtcDechunk* stream = new O2OXtcDechunk( streamFiles, maxDgSize, skipDamaged ) ;
    m_streams.push_back( stream ) ;
    Pds::Dgram* dg = ::read_dg(stream);
    if ( dg ) updateDgramTime( *dg );
    m_dgrams.push_back( dg ) ;

  }

}

//--------------
// Destructor --
//--------------
O2OXtcMerger::~O2OXtcMerger ()
{
  for ( std::vector<O2OXtcDechunk*>::const_iterator it = m_streams.begin() ; it != m_streams.end() ; ++ it ) {
    delete *it ;
  }
}

// read next datagram, return zero pointer after last file has been read,
// throws exception for errors.
Pds::Dgram*
O2OXtcMerger::next()
{
  unsigned ns =  m_streams.size() ;

  // find datagram with lowest timestamp
  int stream = -1 ;
  for ( unsigned i = 0 ; i < ns ; ++ i ) {
    if ( m_dgrams[i] ) {
      if ( stream < 0 or m_dgrams[stream]->seq.clock() > m_dgrams[i]->seq.clock() ) {
        stream = i ;
      }
    }
  }

  MsgLog( logger, debug, "next -- stream: " << stream ) ;

  if ( stream < 0 ) {
    // means no datagrams left
    return 0 ;
  }

  MsgLog( logger, debug, "next -- file: " << m_streams[stream]->chunkName().basename() ) ;

  // make a copy of the datagram
  Pds::Dgram* dg = ::dg_copy( m_dgrams[stream] );

  MsgLog( logger, debug, "next -- m_dgrams[stream].clock: "
      << dg->seq.clock().seconds() << " sec " << dg->seq.clock().nanoseconds() << " nsec" ) ;
  MsgLog( logger, debug, "next -- m_dgrams[stream].service: " << Pds::TransitionId::name(dg->seq.service()) ) ;

  // check the type of the datagram, for L1Accept give it to the caller,
  // all other datagram types must appear in all streams, return only one copy
  if ( dg->seq.service() == Pds::TransitionId::L1Accept ) {

    // get next datagram from that stream
    MsgLog( logger, debug, "next -- read datagram from file: " << m_streams[stream]->chunkName().basename() ) ;
    Pds::Dgram* dg = ::read_dg(m_streams[stream]) ;
    if ( dg ) updateDgramTime( *dg );
    m_dgrams[stream] = dg ;

  } else {

    // make sure that all streams have the same type of datagram
    for ( unsigned i = 0 ; i < ns ; ++ i ) {
      if ( m_dgrams[i] and m_dgrams[i]->seq.service() != dg->seq.service() ) {
        MsgLog( logger, error, "next -- streams desynchronized:"
            << "\n    stream[" << stream << "] = "
            << m_streams[stream]->chunkName().basename()
            << " service = " << Pds::TransitionId::name(dg->seq.service())
            << " time = " << dg->seq.clock().seconds() << " sec " << dg->seq.clock().nanoseconds() << " nsec"
            << " damage = " << std::hex << dg->xtc.damage.value() << std::dec
            << "\n    stream[" << i << "] = "
            << m_streams[i]->chunkName().basename()
            << " service = " << Pds::TransitionId::name(m_dgrams[i]->seq.service())
            << " time = " << m_dgrams[i]->seq.clock().seconds() << " sec " << m_dgrams[i]->seq.clock().nanoseconds() << " nsec"
            << " damage = " << std::hex << m_dgrams[i]->xtc.damage.value() << std::dec
            ) ;
        delete [] (char*)dg ;
        throw O2OXTCSyncException() ;
      }
    }

    // delete all datagrams except the one that we return and read next
    // datagram from every stream
    for ( unsigned i = 0 ; i < ns ; ++ i ) {
      if ( m_dgrams[i] ) {
        Pds::Dgram* dg = ::read_dg(m_streams[i]) ;
        if ( dg ) updateDgramTime( *dg );
        m_dgrams[i] = dg ;
      }
    }

  }

  return dg ;
}


} // namespace O2OTranslator
