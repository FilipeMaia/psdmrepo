//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcOutputModule...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcOutput/XtcOutputModule.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/make_shared.hpp>
#include <stdexcept>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"
#include "PSXtcOutput/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace PSXtcOutput;
PSANA_MODULE_FACTORY(XtcOutputModule)

namespace fs = boost::filesystem;

namespace {

  char logger[] = "XtcOutputModule";

  // special delete for boost::shared_ptr
  struct CharArrayDeleter {
    void operator()(char* p) { delete [] p; }
  };

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSXtcOutput {

//----------------
// Constructors --
//----------------
XtcOutputModule::XtcOutputModule (const std::string& name)
  : Module(name, true)
  , m_chunkSizeMB(0)
  , m_nameFmt()
  , m_dirName()
  , m_keepEpics(true)
  , m_expNum(0)
  , m_run(0)
  , m_stream(0)
  , m_chunk(0)
  , m_fd(-1)
  , m_storedBytes(0)
  , m_filter()
{
  // get the values from configuration or use defaults
  m_chunkSizeMB = config("chunkSizeMB", 512000U);
  //m_nameFmt = configStr("nameFmt", "e%1$d-r%2$04d-s%3$02d-c%4$02d.xtcf");
  m_nameFmt = configStr("nameFmt", "e%1$d-r%2$04d.xtcf");
  m_dirName = configStr("dirName", ".");
  m_keepEpics = config("keepEpics", true);
  m_expNum = config("expNum", 0);
  m_run = config("run", 0);
  m_stream = config("stream", 0);

  // make filter if epics is needed
  if (m_keepEpics) {
    XtcInput::XtcFilterTypeId::IdList keep(1, Pds::TypeId::Id_Epics);
    XtcInput::XtcFilterTypeId::IdList discard;
    m_filter = boost::make_shared<XtcInput::XtcFilter<XtcInput::XtcFilterTypeId> >(XtcInput::XtcFilterTypeId(keep, discard));
  }
}

//--------------
// Destructor --
//--------------
XtcOutputModule::~XtcOutputModule ()
{
  // should be closed already in endJob(), additional protection here
  if (m_fd >= 0) {
    close(m_fd);
  }
}

/// Method which is called once at the beginning of the job
void 
XtcOutputModule::beginJob(Event& evt, Env& env)
{
  // run number should probably be known already
  if (m_run == 0) {
    shared_ptr<EventId> eventId = evt.get();
    if (shared_ptr<EventId> eventId = evt.get()) {
      m_run = eventId->run();
    }
  }
  if (m_expNum == 0) {
    m_expNum = env.expNum();
  }

  // get datagram from event
  if (XtcInput::Dgram::ptr dg = evt.get()) {
    // save its data
    saveData(dg.get(), Pds::TransitionId::Configure);
  }
}

/// Method which is called at the beginning of the run
void 
XtcOutputModule::beginRun(Event& evt, Env& env)
{
  // get datagram from event
  if (XtcInput::Dgram::ptr dg = evt.get()) {
    // save its data
    saveData(dg.get(), Pds::TransitionId::BeginRun);
  }
}

/// Method which is called at the beginning of the calibration cycle
void 
XtcOutputModule::beginCalibCycle(Event& evt, Env& env)
{
  // get datagram from event
  if (XtcInput::Dgram::ptr dg = evt.get()) {
    // save its data
    saveData(dg.get(), Pds::TransitionId::BeginCalibCycle);
  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
XtcOutputModule::event(Event& evt, Env& env)
{
  // get datagram from event
  if (XtcInput::Dgram::ptr dg = evt.get()) {

    // check presence of skip flag
    boost::shared_ptr<int> skip = evt.get("__psana_skip_event__");
    if (not skip) {

      MsgLog(logger, trace, "XtcOutputModule: storing complete event");

      // save its data
      saveData(dg.get(), Pds::TransitionId::L1Accept);

    } else if (m_filter) {

      // filter out everything except EPICS and save

      // make a buffer big enough
      size_t bufSize = dg->xtc.sizeofPayload() + sizeof(Pds::Dgram);
      boost::shared_ptr<char> buf(new char[bufSize], CharArrayDeleter());

      // call filter, if anything left after filtering then save it
      size_t newSize = m_filter->filter(dg.get(), buf.get());
      MsgLog(logger, trace, "XtcOutputModule: filtering event, bytes before = " << bufSize << ", bytes after = " << newSize);
      if (newSize) {
        saveData((Pds::Dgram*)buf.get(), Pds::TransitionId::L1Accept);
      }

    }
  }
}
  
/// Method which is called at the end of the calibration cycle
void 
XtcOutputModule::endCalibCycle(Event& evt, Env& env)
{
  // get datagram from event
  if (XtcInput::Dgram::ptr dg = evt.get()) {
    // save its data
    saveData(dg.get(), Pds::TransitionId::EndCalibCycle);
  }
}

/// Method which is called at the end of the run
void 
XtcOutputModule::endRun(Event& evt, Env& env)
{
  // get datagram from event
  if (XtcInput::Dgram::ptr dg = evt.get()) {
    // save its data
    saveData(dg.get(), Pds::TransitionId::EndRun);
  }
}

/// Method which is called once at the end of the job
void 
XtcOutputModule::endJob(Event& evt, Env& env)
{
  // time to close it
  if (m_fd >= 0) {
    close(m_fd);
    m_fd = -1;
  }
}

/// Method that writes the data to output file, opening and closing it if necessary
void
XtcOutputModule::saveData(Pds::Dgram* dg, Pds::TransitionId::Value transition)
{
  // check that datagram has correct transition type
  if (dg->seq.service() != transition) {
    MsgLog(logger, warning, "XtcOutputModule: datagram has unexpected transition: " << Pds::TransitionId::name(dg->seq.service())
      << ", expected: " << Pds::TransitionId::name(transition) << ", will skip datagram");
    return;
  }

  // may need to close file if limit is reached
  if ((m_storedBytes + 1048576 - 1)/1048576 >= m_chunkSizeMB) {
    close(m_fd);
    m_fd = -1;
    m_storedBytes = 0;
    ++ m_chunk;
  }

  // open file if needed
  if (m_fd < 0) {

    // build the name
    std::string fname;
    try {
      // Ingrid Ofte: I have removed the stream and chunk from the output file name, 
      // because this only a single file... Change it back if needed in the future... 
      //fname = boost::str(boost::format(m_nameFmt) % m_expNum % m_run % m_stream % m_chunk);
      fname = boost::str(boost::format(m_nameFmt) % m_expNum % m_run);
      fs::path path(m_dirName);
      path /= fname;
      fname = path.string();
    } catch (const std::exception& exc) {
      throw FileNameFormatError(ERR_LOC, m_nameFmt, exc.what());
    }

    // open it
    MsgLog(logger, info, "XtcOutputModule: opening new file: " << fname);
    m_fd = creat(fname.c_str(), 0666);
    if (m_fd < 0) throw FileOpenError(ERR_LOC, fname);
  }

  // save the data
  char* buf = (char*)dg;
  size_t size = dg->xtc.sizeofPayload() + sizeof(Pds::Dgram);
  while (size) {
    ssize_t n = write(m_fd, buf, size);
    if (n < 0) {
      if (errno == EAGAIN) {
        // retry
        n = 0;
      } else {
        throw FileWriteError(ERR_LOC);
      }
    }
    size -= n;
    buf += n;
    m_storedBytes += n;
  }

}


} // namespace PSXtcOutput
