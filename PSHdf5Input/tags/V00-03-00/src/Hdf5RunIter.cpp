//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5RunIter...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSHdf5Input/Hdf5RunIter.h"

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
#include "PSHdf5Input/Hdf5CalibCycleIter.h"
#include "PSHdf5Input/Hdf5EventId.h"
#include "PSHdf5Input/Hdf5Utils.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "Hdf5RunIter";
  
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
Hdf5RunIter::Hdf5RunIter (const hdf5pp::Group& grp, int runNumber,
    unsigned schemaVersion, bool fullTsFormat)
  : m_grp(grp)
  , m_runNumber(runNumber)
  , m_schemaVersion(schemaVersion)
  , m_fullTsFormat(fullTsFormat)
  , m_groups()
  , m_ccIter()
{
  // get all subgroups which start with 'CalibCycle:'
  hdf5pp::GroupIter giter(m_grp);
  for (hdf5pp::Group grp = giter.next(); grp.valid(); grp = giter.next()) {
    const std::string& grpname = grp.basename();
    if (grpname == "CalibCycle" or boost::algorithm::starts_with(grpname, "CalibCycle:")) {
      m_groups.push_back(grp);
    }    
  }

  // sort them by name
  m_groups.sort(GroupCmp());
}

//--------------
// Destructor --
//--------------
Hdf5RunIter::~Hdf5RunIter ()
{
}

// Returns next object
Hdf5RunIter::value_type 
Hdf5RunIter::next()
{
  Hdf5IterData res;

  if (not m_ccIter.get()) {
    
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
      m_ccIter.reset(new Hdf5CalibCycleIter(grp, m_runNumber, m_schemaVersion, m_fullTsFormat));

      // make event id
      PSTime::Time etime = Hdf5Utils::getTime(m_ccIter->group(), "start");
      boost::shared_ptr<PSEvt::EventId> eid;
      if (etime != PSTime::Time(0,0)) eid = boost::make_shared<Hdf5EventId>(m_runNumber, etime, 0x1ffff, 0, 0,0);
      res = Hdf5IterData(Hdf5IterData::BeginCalibCycle, eid);

      // fill result with the configuration object data locations
      hdf5pp::GroupIter giter(grp);
      for (hdf5pp::Group grp1 = giter.next(); grp1.valid(); grp1 = giter.next()) {
        if (grp1.basename() != "Epics::EpicsPv") {
          hdf5pp::GroupIter subgiter(grp1);
          for (hdf5pp::Group grp2 = subgiter.next(); grp2.valid(); grp2 = subgiter.next()) {
            if (not grp2.hasChild("time")) {
              res.add(grp2, -1);
            }
          }
        }
      }

    }

  } else {

    // read next event from this iterator
    res = m_ccIter->next();
  
    // switch to next group if it sends us Stop
    if (res.type() == value_type::Stop) {

      PSTime::Time etime = Hdf5Utils::getTime(m_ccIter->group(), "end");
      boost::shared_ptr<PSEvt::EventId> eid;
      if (etime != PSTime::Time(0,0)) eid = boost::make_shared<Hdf5EventId>(m_runNumber, etime, 0x1ffff, 0, 0,0);
      res = Hdf5IterData(Hdf5IterData::EndCalibCycle, eid);

      m_ccIter.reset();

    }

  }

  return res;
}

} // namespace PSHdf5Input
