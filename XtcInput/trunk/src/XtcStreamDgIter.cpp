//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcStreamDgIter...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/XtcStreamDgIter.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "XtcInput/Exceptions.h"
#include "XtcInput/XtcChunkDgIter.h"
#include "pdsdata/xtc/Xtc.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "XtcStreamDgIter" ;

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
XtcStreamDgIter::XtcStreamDgIter(const boost::shared_ptr<ChunkFileIterI>& chunkIter, size_t maxDgSize, bool skipDamaged )
  : m_chunkIter(chunkIter)
  , m_maxDgSize(maxDgSize)
  , m_skipDamaged(skipDamaged)
  , m_file()
  , m_dgiter()
  , m_count(0)
{
}

//--------------
// Destructor --
//--------------
XtcStreamDgIter::~XtcStreamDgIter ()
{
}

// read next datagram, return zero pointer after last file has been read,
// throws exception for errors.
Dgram::ptr
XtcStreamDgIter::next()
{
  Dgram::ptr dgram;
  while (not dgram) {

    if (not m_dgiter) {

      // get next file name
      m_file = m_chunkIter->next();

      // if no more file then stop
      if (m_file.path().empty()) break ;

      // open next xtc file if there is none open
      MsgLog(logger, trace, "processing file: " << m_file) ;
      m_dgiter = boost::make_shared<XtcChunkDgIter>(m_file.path(), m_maxDgSize, m_chunkIter->liveTimeout());
      m_count = 0 ;
    }

    // try to read next event from it
    dgram = m_dgiter->next() ;
    ++ m_count ;

    // if failed to read go to next file
    if ( not dgram.get() ) {
      m_dgiter.reset();
    } else if ( m_skipDamaged ) {

      // get rid of damaged datagrams
      const Pds::Xtc& xtc = dgram->xtc ;
      if ( xtc.damage.value() ) {
        MsgLog( logger, warning, "XTC damage: " << std::hex << xtc.damage.value() << std::dec
            << " level: " << int(xtc.src.level()) << '#' << Pds::Level::name(xtc.src.level())
            << " type: " << int(xtc.contains.id()) << '#' << Pds::TypeId::name(xtc.contains.id())
            << "/V" << xtc.contains.version()
            << "\n    Skipping damaged event -- file: " << m_file << " event: " << m_count ) ;
        dgram.reset();
      }

    }

  }

  return dgram ;
}

// get current file name
XtcFileName
XtcStreamDgIter::chunkName() const
{
  return m_file ;
}

} // namespace XtcInput
