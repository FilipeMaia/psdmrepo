//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcInputModule...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcInput/XtcInputModule.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <iterator>
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/epics.ddl.h"
#include "PSTime/Time.h"
#include "PSXtcInput/Exceptions.h"
#include "PSXtcInput/XtcEventId.h"
#include "XtcInput/DgramQueue.h"
#include "XtcInput/DgramReader.h"
#include "XtcInput/XtcFileName.h"
#include "XtcInput/XtcIterator.h"
#include "XtcInput/MergeMode.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace XtcInput;

using namespace PSXtcInput;
PSANA_INPUT_MODULE_FACTORY(XtcInputModule)


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSXtcInput {

//----------------
// Constructors --
//----------------
XtcInputModule::XtcInputModule (const std::string& name)
  : XtcInputModuleBase(name)
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
XtcInputModule::~XtcInputModule ()
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

// Initialization method for external datagram source
void 
XtcInputModule::initDgramSource()
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

// Get the next datagram from some external source
XtcInput::Dgram
XtcInputModule::nextDgram()
{
  return m_dgQueue->pop();
}

} // namespace PSXtcInput
