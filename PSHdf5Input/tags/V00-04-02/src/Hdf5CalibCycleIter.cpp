//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5CalibCycleIter...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSHdf5Input/Hdf5CalibCycleIter.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <iterator>
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/GroupIter.h"
#include "PSHdf5Input/Hdf5EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSHdf5Input {

//----------------
// Constructors --
//----------------
Hdf5CalibCycleIter::Hdf5CalibCycleIter (const hdf5pp::Group& calibCycleGrp, int runNumber,
    unsigned schemaVersion, bool fullTsFormat)
  : m_calibCycleGrp(calibCycleGrp)
  , m_runNumber(runNumber)
  , m_schemaVersion(schemaVersion)
  , m_fullTsFormat(fullTsFormat)
  , m_merger()
{

  // find all groups with the event data (those which have "time" dataset)
  hdf5pp::GroupIter typeIter(calibCycleGrp, hdf5pp::GroupIter::HardLink);
  for (hdf5pp::Group typeGrp = typeIter.next(); typeGrp.valid(); typeGrp = typeIter.next()) {
    hdf5pp::GroupIter srcIter(typeGrp, hdf5pp::GroupIter::HardLink);
    
    // Epics group is 3-level deep, regular groups are 2-level deep
    if (typeGrp.basename() == "Epics::EpicsPv") {
      
      for (hdf5pp::Group epicsSrcGrp = srcIter.next(); 
           epicsSrcGrp.valid(); 
           epicsSrcGrp = srcIter.next()) {
        hdf5pp::GroupIter epicsPvNameIter(epicsSrcGrp, hdf5pp::GroupIter::HardLink);
        for (hdf5pp::Group epicsPvNameGrp = epicsPvNameIter.next(); 
             epicsPvNameGrp.valid(); 
             epicsPvNameGrp = epicsPvNameIter.next()) {
          if (epicsPvNameGrp.hasChild("time")) {
            Hdf5DatasetIter begin(epicsPvNameGrp, m_fullTsFormat);
            Hdf5DatasetIter end(epicsPvNameGrp, m_fullTsFormat, Hdf5DatasetIter::End);
            m_merger.add(begin, end);
          }
        }
      }

    } else {
      
      for (hdf5pp::Group srcGrp = srcIter.next(); srcGrp.valid(); srcGrp = srcIter.next()) {
        if (srcGrp.hasChild("time")) {
          Hdf5DatasetIter begin(srcGrp, m_fullTsFormat);
          Hdf5DatasetIter end(srcGrp, m_fullTsFormat, Hdf5DatasetIter::End);
          m_merger.add(begin, end);
        }
      }
    }
  }

}

//--------------
// Destructor --
//--------------
Hdf5CalibCycleIter::~Hdf5CalibCycleIter ()
{
}

// Returns next object
Hdf5CalibCycleIter::value_type 
Hdf5CalibCycleIter::next()
{
  typedef std::vector<Hdf5DatasetIterData> MergedData;
  MergedData data;
  data.reserve(m_merger.size());
  
  value_type res;

  if (not m_merger.next(std::back_inserter(data))) {

    res = value_type(value_type::Stop, boost::shared_ptr<PSEvt::EventId>());

  } else {

    boost::shared_ptr<Hdf5EventId> eid;

    if (not data.empty()) {
      // make EventId out first item
      const Hdf5DatasetIterData& d = data.front();
      PSTime::Time time(d.sec, d.nsec);
      eid = boost::make_shared<Hdf5EventId>(m_runNumber, time, d.fiducials, d.ticks, d.vector, d.control);
    }

    res = Hdf5IterData(value_type::Event, eid);
    for (MergedData::const_iterator it = data.begin(); it != data.end(); ++ it) {
      res.add(it->group, it->index, it->mask);
    }
  }

  return res;
}

} // namespace PSHdf5Input
