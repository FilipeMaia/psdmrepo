//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpEpics...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpEpics.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <algorithm>
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpEpics)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpEpics::DumpEpics (const std::string& name)
  : Module(name)
{
}

//--------------
// Destructor --
//--------------
DumpEpics::~DumpEpics ()
{
}

/// Method which is called at the beginning of the calibration cycle
void
DumpEpics::beginCalibCycle(Event& evt, Env& env)
{
  const EpicsStore& estore = env.epicsStore();

  // Print the list of aliases
  const std::vector<std::string>& aliases = estore.aliases();
  WithMsgLog(name(), info, str) {
    str << "Total number of EPICS Aliases: " << aliases.size();
    for (std::vector<std::string>::const_iterator it = aliases.begin(); it != aliases.end(); ++ it) {
      str << "\n  '" << *it << "' -> '" << estore.pvName(*it) << "'";
    }
  }

  // and dump all values as well
  event(evt, env);
}

// Method which is called with event data
void 
DumpEpics::event(Event& evt, Env& env)
{
  // access EPICS store
  const EpicsStore& estore = env.epicsStore();
  
  // get the names of EPICS PVs
  std::vector<std::string> pvNames = estore.pvNames();
  std::sort(pvNames.begin(), pvNames.end());
  size_t size = pvNames.size();
  
  std::stringstream str;

  str << "Total number of EPICS PVs: " << pvNames.size() << '\n';

  for (size_t i = 0; i < size; ++ i) {
      
   // get generic PV object, only useful if you want to access
    // its type, and array size
    shared_ptr<Psana::Epics::EpicsPvHeader> pv = estore.getPV(pvNames[i]);
    
    // print generic info
    str << "  " << pvNames[i] << " id=" << pv->pvId()
        << " type=" << pv->dbrType()  
        << " isCtrl=" << int(pv->isCtrl())
        << " isTime=" << int(pv->isTime())
        << " size=" << pv->numElements() << '\n';
      
    // print status info
    int status, severity;
    PSTime::Time time;
    estore.status(pvNames[i], status, severity, time);
    str << "    status=" << status << ", severity=" << severity 
        << " time=" << time << '\n';
    
    // print all values. Either use simpler interface of the 
    // epicsStore value function, or for demonstration purposes, get a 
    // pointer to the actual epics type. Using the value function is
    // generally simplest, but when working with epics pv's that are 
    // waveforms, getting the actual type may be simpler.
    
    str << "    values: ";
    
    std::string dataArrayStr;
    if (pv->numElements() > 1) {
      dataArrayStr = dumpPvDataArray(estore, pvNames[i]);
    }

    if (dataArrayStr.size()>0) {
      str << dataArrayStr;
    } else {
      // demo simpler way to get values, through estore.value with pvName lookup:
      for (int e = 0; e < pv->numElements(); ++ e) {
        // get value and convert to string 
        // if value is numeric, could also assign it to appropriate
        // numberic types, int, double, float, etc)
        const std::string& value = estore.value(pvNames[i], e);
        str << ' ' << value;
      }
    }
    str << std::endl;
  }
  MsgLog(name(), info, str.str());
}

//		----------------------------------------
// 		-- Protected Function Member Definitions --
//		----------------------------------------

std::string  DumpEpics::dumpPvDataArray(const PSEnv::EpicsStore &estore, const std::string &pvName) {
  // Since we don't know the type of data, we'll try a few (see psddl_psana/include/epics.ddl.h for all types)
  // An alternative, below, is to switch on the dbrType
  std::stringstream str;
  boost::shared_ptr<Psana::Epics::EpicsPvTimeShort> pvTimeShort = estore.getPV(pvName);
  if (pvTimeShort) {
    ndarray<const int16_t, 1> data = pvTimeShort->data();
    str << data;
  } else {
    boost::shared_ptr<Psana::Epics::EpicsPvTimeFloat> pvTimeFloat = estore.getPV(pvName);
    if (pvTimeFloat) {
      ndarray<const float, 1> data = pvTimeFloat->data();
      str << data;
    } else {
      boost::shared_ptr<Psana::Epics::EpicsPvCtrlShort> pvCtrlShort = estore.getPV(pvName);
      if (pvCtrlShort) {
        ndarray<const int16_t, 1> data = pvCtrlShort->data();
        str << data;
      } else {
        boost::shared_ptr<Psana::Epics::EpicsPvCtrlFloat> pvCtrlFloat = estore.getPV(pvName);
        if (pvCtrlFloat) {
          ndarray<const float, 1> data = pvCtrlFloat->data();
          str << data;
        }
      }
    }
  }

  // demo switching on the dbr type. see psddl_psana/include/epics.ddl.h for all DBR types
  if (str.str().size() == 0) {
    shared_ptr<Psana::Epics::EpicsPvHeader> pv = estore.getPV(pvName);
    if (pv) {
      switch (pv->dbrType()) {
      case Psana::Epics::DBR_TIME_DOUBLE:
        {
          boost::shared_ptr<Psana::Epics::EpicsPvTimeDouble> pvTimeDouble = estore.getPV(pvName);
          if (not pvTimeDouble) {
            MsgLog(name(), error, "unexpected: dbrType/psana type mismatch");
            break;
          }
          ndarray<const double, 1> data = pvTimeDouble->data();
          str << data;
        }
        break;

      case Psana::Epics::DBR_CTRL_DOUBLE:
        {
          boost::shared_ptr<Psana::Epics::EpicsPvCtrlDouble> pvCtrlDouble = estore.getPV(pvName);
          if (not pvCtrlDouble) {
            MsgLog(name(), error, "unexpected: dbrType/psana type mismatch");
            break;
          }
          ndarray<const double, 1> data = pvCtrlDouble->data();
          str << data;
        }
        break;
      }
    }
  }
  
  return str.str();
}


  
} // namespace psana_examples
