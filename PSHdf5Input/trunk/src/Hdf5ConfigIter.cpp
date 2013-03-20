//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5ConfigIter...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSHdf5Input/Hdf5ConfigIter.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/make_shared.hpp>
#include <boost/algorithm/string/predicate.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/GroupIter.h"
#include "PSHdf5Input/Exceptions.h"
#include "PSHdf5Input/Hdf5EventId.h"
#include "PSHdf5Input/Hdf5RunIter.h"
#include "PSHdf5Input/Hdf5Utils.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "Hdf5ConfigIter";
  
  // comparison operator for groups
  struct GroupCmp {
    bool operator()(const hdf5pp::Group& lhs, const hdf5pp::Group& rhs) const {
      return lhs.name() < rhs.name();
    }
  };
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSHdf5Input {

//----------------
// Constructors --
//----------------
Hdf5ConfigIter::Hdf5ConfigIter (const hdf5pp::Group& grp, int runNumber,
    unsigned schemaVersion, bool fullTsFormat)
  : m_grp(grp)
  , m_runNumber(runNumber)
  , m_schemaVersion(schemaVersion)
  , m_fullTsFormat(fullTsFormat)
  , m_groups()
  , m_runIter()
{
  // get all subgroups which start with 'Run:'
  hdf5pp::GroupIter giter(m_grp);
  for (hdf5pp::Group grp = giter.next(); grp.valid(); grp = giter.next()) {
    const std::string& grpname = grp.basename();
    if (grpname == "Run" or boost::algorithm::starts_with(grpname, "Run:")) {
      m_groups.push_back(grp);
    }    
  }

  // sort them by name
  m_groups.sort(GroupCmp());
}

//--------------
// Destructor --
//--------------
Hdf5ConfigIter::~Hdf5ConfigIter ()
{
}

// Returns next object
Hdf5ConfigIter::value_type 
Hdf5ConfigIter::next()
{
  Hdf5IterData res;

  if (not m_runIter.get()) {
    
    // no more run groups left - we are done
    if (m_groups.empty()) {

      MsgLog(logger, debug, "stop iterating in group: " << m_grp.name());
      res = value_type(value_type::Stop, boost::shared_ptr<PSEvt::EventId>());

    } else {

      // open next group
      hdf5pp::Group grp = m_groups.front();
      m_groups.pop_front();
      MsgLog(logger, debug, "switching to group: " << grp.name());

      // make iter for this new group
      m_runIter.reset(new Hdf5RunIter(grp, m_runNumber, m_schemaVersion, m_fullTsFormat));

      PSTime::Time etime = Hdf5Utils::getTime(m_runIter->group(), "start");
      boost::shared_ptr<PSEvt::EventId> eid = boost::make_shared<Hdf5EventId>(m_runNumber, etime, 0x1ffff, 0, 0);
      res = Hdf5IterData(Hdf5IterData::BeginRun, eid);

    }

  } else {

    // read next event from this iterator
    res = m_runIter->next();

    // switch to next group if it sends us Stop
    if (res.type() == value_type::Stop) {
  
      PSTime::Time etime = Hdf5Utils::getTime(m_runIter->group(), "end");
      boost::shared_ptr<PSEvt::EventId> eid = boost::make_shared<Hdf5EventId>(m_runNumber, etime, 0x1ffff, 0, 0);
      res = Hdf5IterData(Hdf5IterData::EndRun, eid);

      m_runIter.reset();
    }
  }

  return res;

}

} // namespace PSHdf5Input
