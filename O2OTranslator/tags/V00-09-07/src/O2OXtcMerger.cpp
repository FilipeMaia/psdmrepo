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
    Pds::Dgram* dg = stream->next();
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

  // get next datagram from that stream
  MsgLog( logger, debug, "next -- read datagram from file: " << m_streams[stream]->chunkName().basename() ) ;
  Pds::Dgram* ndg = m_streams[stream]->next();
  if ( ndg ) updateDgramTime( *ndg );
  m_dgrams[stream] = ndg ;

  return dg ;
}


void 
O2OXtcMerger::updateDgramTime(Pds::Dgram& dgram) const
{
  if ( dgram.seq.service() != Pds::TransitionId::L1Accept ) {

    // update clock values
    const Pds::ClockTime& time = dgram.seq.clock() ;
    int32_t sec = time.seconds() + m_l1OffsetSec;
    int32_t nsec = time.nanoseconds() + m_l1OffsetNsec;
    if (nsec < 0) {
        nsec += 1000000000;
        -- sec;
    } else if (nsec >= 1000000000) {
        nsec -= 1000000000;
        ++ sec;
    }      
    Pds::ClockTime newTime(sec, nsec) ;

    // there is no way to change clock field in datagram but there is 
    // an assignment operator
    dgram.seq = Pds::Sequence(newTime, dgram.seq.stamp());
  }
}

} // namespace O2OTranslator
