//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5FileIter...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSHdf5Input/Hdf5FileIter.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/make_shared.hpp>
#include <boost/algorithm/string/predicate.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/GroupIter.h"
#include "hdf5pp/PListFileAccess.h"
#include "PSHdf5Input/Exceptions.h"
#include "PSHdf5Input/Hdf5ConfigIter.h"
#include "PSHdf5Input/Hdf5EventId.h"
#include "PSHdf5Input/Hdf5Utils.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "Hdf5FileIter";
  
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
Hdf5FileIter::Hdf5FileIter (const std::string& fileName)
  : m_fileName(fileName)
  , m_file()
  , m_groups()
  , m_configIter()
  , m_runNumber()
  , m_schemaVersion(0)
  , m_fullTsFormat(false)
{
  // Define file access properties.
  // Do not want to cache too many chunks, but for some types chunks
  // can be very big, up to 100MB. These numbers are per dataset.
  hdf5pp::PListFileAccess fapl;
  size_t rdcc_nelmts = 5;
  size_t rdcc_nbytes = 101*1024*1024;  
  fapl.set_cache(rdcc_nelmts, rdcc_nbytes, 0.9);

  // open a file
  try {
    MsgLog(logger, debug, "opening new file: " << m_fileName);
    m_file = hdf5pp::File::open(m_fileName, hdf5pp::File::Read, fapl);
  } catch (const hdf5pp::Exception& ex) {
    throw FileOpenError(ERR_LOC, m_fileName, ex.what());
  }
  
  // open top group
  hdf5pp::Group top = m_file.openGroup("/");

  // read some attributes
  if (top.hasAttr(":schema:version")) m_schemaVersion = top.openAttr<uint32_t>(":schema:version").read();
  if (top.hasAttr(":schema:timestamp-format")) {
    std::string tsFmt = top.openAttr<const char*>(":schema:timestamp-format").read();
    m_fullTsFormat = tsFmt == "full";
  }

  // get all subgroups which start with 'Configure:'
  hdf5pp::GroupIter giter(top);
  for (hdf5pp::Group grp = giter.next(); grp.valid(); grp = giter.next()) {
    const std::string& grpname = grp.basename();
    if (grpname == "Configure" or boost::algorithm::starts_with(grpname, "Configure:")) {
      m_groups.push_back(grp);
    }    
  }

  // sort them by name
  m_groups.sort(GroupCmp());
  
  // might have run number attribute
  if (top.hasAttr("runNumber")) {
    m_runNumber = top.openAttr<int>("runNumber").read();
    MsgLog(logger, debug, "file defines run number: " << m_runNumber);
  }
}

//--------------
// Destructor --
//--------------
Hdf5FileIter::~Hdf5FileIter ()
{
}

// Returns next object
Hdf5FileIter::value_type 
Hdf5FileIter::next()
{
  Hdf5IterData res;

  if (not m_configIter) {

    // no more config groups left - we are done
    if (m_groups.empty()) {
      MsgLog(logger, debug, "stop iterating in file: " << m_fileName);
      res = value_type(value_type::Stop, boost::shared_ptr<PSEvt::EventId>());
    } else {

      // open next group
      hdf5pp::Group grp = m_groups.front();
      m_groups.pop_front();
      MsgLog(logger, debug, "switching to group: " << grp.name());

      // make iter for this new group
      m_configIter.reset(new Hdf5ConfigIter(grp, m_runNumber, m_schemaVersion, m_fullTsFormat));

      PSTime::Time etime = Hdf5Utils::getTime(m_configIter->group(), "start");
      boost::shared_ptr<PSEvt::EventId> eid;
      if (etime != PSTime::Time(0,0)) eid = boost::make_shared<Hdf5EventId>(m_runNumber, etime, 0x1ffff, 0, 0,0);
      res = Hdf5IterData(Hdf5IterData::Configure, eid);
      
      // fill result with the configuration object data locations
      hdf5pp::GroupIter giter(grp);
      for (hdf5pp::Group grp1 = giter.next(); grp1.valid(); grp1 = giter.next()) {
        const std::string& grpname = grp1.basename();

        // Epics group is 3-level deep, regular groups are 2-level deep
        if (grpname == "Epics::EpicsPv") {

          hdf5pp::GroupIter subgiter(grp1);
          for (hdf5pp::Group grp2 = subgiter.next(); grp2.valid(); grp2 = subgiter.next()) {
            // store this group so that converter can find aliases
            res.add(grp2, -1);
            // Do not include soft links
            hdf5pp::GroupIter subgiter2(grp2, hdf5pp::GroupIter::HardLink);
            for (hdf5pp::Group grp3 = subgiter2.next(); grp3.valid(); grp3 = subgiter2.next()) {
              res.add(grp3, 0);
            }
          }

        } else if (grpname != "Run" and not boost::algorithm::starts_with(grpname, "Run:")) {

          hdf5pp::GroupIter subgiter(grp1);
          for (hdf5pp::Group grp2 = subgiter.next(); grp2.valid(); grp2 = subgiter.next()) {
            if (grp2.hasChild("time")) {
              res.add(grp2, 0);
            } else {
              res.add(grp2, -1);
            }
          }

        }
      }

    }
    
  } else {

    // read next event from this iterator
    res = m_configIter->next();
  
    // switch to next group if it sends us Stop
    if (res.type() == value_type::Stop) {

      PSTime::Time etime = Hdf5Utils::getTime(m_configIter->group(), "end");
      boost::shared_ptr<PSEvt::EventId> eid;
      if (etime != PSTime::Time(0,0)) eid = boost::make_shared<Hdf5EventId>(m_runNumber, etime, 0x1ffff, 0, 0, 0);
      res = Hdf5IterData(Hdf5IterData::UnConfigure, eid);

      m_configIter.reset();

    }

  }
  
  return res;

}

} // namespace PSHdf5Input
