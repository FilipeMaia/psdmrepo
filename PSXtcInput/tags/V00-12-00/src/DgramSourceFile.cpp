//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DgramSourceFile...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcInput/DgramSourceFile.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSXtcInput/Exceptions.h"
#include "XtcInput/DgramQueue.h"
#include "XtcInput/DgramReader.h"
#include "XtcInput/MergeMode.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace XtcInput;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSXtcInput {

//----------------
// Constructors --
//----------------
DgramSourceFile::DgramSourceFile (const std::string& name)
  : IDatagramSource()
  , psana::Configurable(name)
  , m_dgQueue(new XtcInput::DgramQueue(10))
  , m_readerThread()
  , m_fileNames()
{
  m_fileNames = configList("files");
  if ( m_fileNames.empty() ) {
    throw EmptyFileList(ERR_LOC);
  }
}

//--------------
// Destructor --
//--------------
DgramSourceFile::~DgramSourceFile ()
{
  if (m_readerThread) {
    // ask the thread to stop
    m_readerThread->interrupt();
    MsgLog(name(), debug, "wait for reader thread to finish");
    // wait until it does
    m_readerThread->join();
    MsgLog(name(), debug, "reader thread has finished");
  }
}

// Initialization method for datagram source
void 
DgramSourceFile::init()
{
  // will throw if no files were defined in config
  WithMsgLog(name(), debug, str) {
    str << "Input files: ";
    std::copy(m_fileNames.begin(), m_fileNames.end(), std::ostream_iterator<std::string>(str, " "));
  }
  
  // start reader thread
  std::string liveDbConn = configStr("liveDbConn", "");
  std::string liveTable = configStr("liveTable", "file");
  unsigned liveTimeout = config("liveTimeout", 120U);
  double l1offset = config("l1offset", 0.0);
  MergeMode merge = mergeMode(configStr("mergeMode", "FileName"));
  m_readerThread.reset( new boost::thread( DgramReader ( m_fileNames.begin(), m_fileNames.end(),
      *m_dgQueue, merge, liveDbConn, liveTable, liveTimeout, l1offset) ) );
}


//  This method returns next datagram from the source
bool
DgramSourceFile::next(std::vector<XtcInput::Dgram>& eventDg, std::vector<XtcInput::Dgram>& nonEventDg)
{
  XtcInput::Dgram dg = m_dgQueue->pop();
  if (not dg.empty()) {
    eventDg.push_back(dg);
    return true;
  } else {
    return false;
  }
}

} // namespace PSXtcInput
