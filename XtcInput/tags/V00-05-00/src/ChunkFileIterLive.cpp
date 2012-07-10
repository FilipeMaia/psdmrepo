//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ChunkFileIterLive...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/ChunkFileIterLive.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "ChunkFileIterLive";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
ChunkFileIterLive::ChunkFileIterLive(unsigned expNum, unsigned run, unsigned stream,
    unsigned liveTimeout, const boost::shared_ptr<LiveFilesDB>& filesdb)
  : ChunkFileIterI()
  , m_expNum(expNum)
  , m_run(run)
  , m_stream(stream)
  , m_liveTimeout(liveTimeout)
  , m_filesdb(filesdb)
  , m_chunk(-1)
{
}

//--------------
// Destructor --
//--------------
ChunkFileIterLive::~ChunkFileIterLive ()
{
}

/**
 *  @brief Return file name for next chunk.
 *
 *  Returns empty name after the last chunk.
 */
XtcFileName
ChunkFileIterLive::next()
{
  // advance to next chunk
  ++ m_chunk;

  XtcFileName fname;

  // See if it is in the database
  std::vector<XtcFileName> files = m_filesdb->files(m_expNum, m_run);
  for (std::vector<XtcFileName>::const_iterator it = files.begin(); it != files.end(); ++ it) {
    if (it->stream() == m_stream and int(it->chunk()) == m_chunk) {
      fname = *it;
      break;
    }
  }

  if (not fname.path().empty()) {

    MsgLog(logger, debug, "Found file in database: " << fname.path());

    // wait until at appears on disk
    std::time_t t0 = std::time(0);
    bool found = false;
    while (not found) {
      if (access(fname.path().c_str(), R_OK) == 0) {
        MsgLog(logger, debug, "Found file on disk: " << fname.path());
        found = true;
        break;
      }
      if (std::time(0) > t0 + m_liveTimeout) break;
      // sleep for one second and repeat
      if (not found) sleep(1);
    }

    // still not found?
    if (not found) {
      MsgLog(logger, warning, "File " << fname.path() << " did not appear on disk after timeout");
      fname = XtcFileName();
    }

  }

  return fname;
}

/**
 *  @brief Return live timeout value
 */
unsigned
ChunkFileIterLive::liveTimeout() const
{
  return m_liveTimeout;
}

} // namespace XtcInput
