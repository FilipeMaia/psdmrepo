//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5InputModule...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSHdf5Input/Hdf5InputModule.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <list>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "IData/Dataset.h"
#include "MsgLogger/MsgLogger.h"
#include "PSHdf5Input/Exceptions.h"
#include "PSHdf5Input/Hdf5EventId.h"
#include "PSHdf5Input/Hdf5FileListIter.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------


using namespace PSHdf5Input;
PSANA_INPUT_MODULE_FACTORY(Hdf5InputModule)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSHdf5Input {

//----------------
// Constructors --
//----------------
Hdf5InputModule::Hdf5InputModule (const std::string& name)
  : psana::InputModule(name)
  , m_datasets()
  , m_iter()
  , m_cvt()
  , m_skipEvents(0)
  , m_maxEvents(0)
  , m_l1Count(0)
{
  // get number of events to process/skip from psana configuration
  ConfigSvc::ConfigSvc cfg;
  m_skipEvents = cfg.get("psana", "skip-events", 0UL);
  m_maxEvents = cfg.get("psana", "events", 0UL);
  m_datasets = configList("files");
  if ( m_datasets.empty() ) {
    throw EmptyFileList(ERR_LOC);
  }
  
  WithMsgLog(this->name(), debug, str) {
    str << "Input datasets: ";
    std::copy(m_datasets.begin(), m_datasets.end(), std::ostream_iterator<std::string>(str, " "));
  }
}

//--------------
// Destructor --
//--------------
Hdf5InputModule::~Hdf5InputModule ()
{
}


// Method which is called once at the beginning of the job
void 
Hdf5InputModule::beginJob(Event& evt, Env& env)
{
  MsgLog(name(), debug, name() << ": in beginJob()");

  // make the list of files from the dataset names
  std::list<std::string> files;
  for (std::vector<std::string>::const_iterator dsiter = m_datasets.begin(); dsiter != m_datasets.end(); ++ dsiter) {
    
    IData::Dataset ds(*dsiter);
    
    // check that dataset is an HDF5 dataset
    if (not ds.exists("h5")) {
      throw NotHdf5Dataset(ERR_LOC, *dsiter);
    }

    const IData::Dataset::NameList& strfiles = ds.files();
    for (IData::Dataset::NameList::const_iterator it = strfiles.begin(); it != strfiles.end(); ++ it) {
      MsgLog(name(), debug, "Hdf5InputModule::beginJob -- add file: " << *it);
      files.push_back(*it);
    }
    
  }
  
  // make iterator
  m_iter.reset(new Hdf5FileListIter(files));
  
  // At the beginJob fill environment with configuration data 
  // from the first configure transition of the first file   
  Hdf5IterData data = m_iter->next();
  MsgLog(name(), debug, "First data item: " << data);

  if (data.type() != Hdf5IterData::Configure) {
    throw FileStructure(ERR_LOC, "Non-configure data at the beginning of file");
  }

  // fill everything
  fillConfig(data, env);
  fillEventId(data, evt);
  fillEpics(data, env);
}

// Method which is called with event data
InputModule::Status 
Hdf5InputModule::event(Event& evt, Env& env)
{
  Hdf5IterData data = m_iter->next();
  MsgLog(name(), debug, "Hdf5InputModule::event -- data: " << data);

  InputModule::Status ret = InputModule::Abort;
  switch(data.type()) {
  case Hdf5IterData::Configure:
    fillConfig(data, env);
    fillEventId(data, evt);
    fillEpics(data, env);
    ret = InputModule::Skip;
    break;
  case Hdf5IterData::BeginRun:
    fillEventId(data, evt);
    ret = InputModule::BeginRun;
    break;
  case Hdf5IterData::BeginCalibCycle:
    fillConfig(data, env);
    fillEventId(data, evt);
    ret = InputModule::BeginCalibCycle;
    break;
  case Hdf5IterData::Event:
    if (m_maxEvents and m_l1Count >= m_skipEvents+m_maxEvents) {
      ret = InputModule::Stop;
    } else if (m_l1Count < m_skipEvents) {
      ret = InputModule::Skip;
    } else {
      fillEventId(data, evt);
      fillEvent(data, evt, env);
      fillEpics(data, env);
      ret = InputModule::DoEvent;
    }
    ++m_l1Count;
    break;
  case Hdf5IterData::EndCalibCycle:
    fillEventId(data, evt);
    m_cvt.resetCache();
    ret = InputModule::EndCalibCycle;
    break;
  case Hdf5IterData::EndRun:
    fillEventId(data, evt);
    m_cvt.resetCache();
    ret = InputModule::EndRun;
    break;
  case Hdf5IterData::UnConfigure:
    fillEventId(data, evt);
    m_cvt.resetCache();
    ret = InputModule::Skip;
    break;
  case Hdf5IterData::Stop:
    m_cvt.resetCache();
    ret = InputModule::Stop;
    break;
  }
  return ret;
}

// Method which is called once at the end of the job
void 
Hdf5InputModule::endJob(Event& evt, Env& env)
{
    
}

// Store config object in environment
void
Hdf5InputModule::fillConfig(const Hdf5IterData& data, Env& env)
{
  MsgLog(name(), debug, name() << ": in fillConfig()");

  // call converter for every piece of data
  const Hdf5IterData::seq_type& pieces = data.data();
  for (Hdf5IterData::const_iterator it = pieces.begin(); it != pieces.end(); ++ it) {
    m_cvt.convertConfig(it->group, it->index, env.configStore());
  }
}

// Store EPICS data in environment
void
Hdf5InputModule::fillEpics(const Hdf5IterData& data, Env& env)
{
  MsgLog(name(), debug, name() << ": in fillEpics()");

  // call converter for every piece of data
  const Hdf5IterData::seq_type& pieces = data.data();
  for (Hdf5IterData::const_iterator it = pieces.begin(); it != pieces.end(); ++ it) {
    m_cvt.convertEpics(it->group, it->index, env.epicsStore());
  }
}

// Store event ID object
void
Hdf5InputModule::fillEventId(const Hdf5IterData& data, Event& evt)
{
  MsgLog(name(), debug, name() << ": in fillEventId()");

  // Store event ID
  boost::shared_ptr<PSEvt::EventId> eventId( new Hdf5EventId(data.run(), data.time()) );
  evt.put(eventId);
}

// Store event data objects
void
Hdf5InputModule::fillEvent(const Hdf5IterData& data, Event& evt, Env& env)
{
  MsgLog(name(), debug, name() << ": in fillEvent()");

  // call converter for every piece of data
  const Hdf5IterData::seq_type& pieces = data.data();
  for (Hdf5IterData::const_iterator it = pieces.begin(); it != pieces.end(); ++ it) {
    m_cvt.convert(it->group, it->index, evt, env.configStore());
  }
}

} // namespace PSHdf5Input
