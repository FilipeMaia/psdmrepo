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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/GroupIter.h"

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
Hdf5CalibCycleIter::Hdf5CalibCycleIter (const hdf5pp::Group& grp)
  : m_grp(grp)
  , m_merger()
{

  // find all groups with the event data (those which have "time" dataset)
  hdf5pp::GroupIter giter(grp);
  for (hdf5pp::Group grp1 = giter.next(); grp1.valid(); grp1 = giter.next()) {
    hdf5pp::GroupIter subgiter(grp1);
    
    // Epics group is 3-level deep, regular groups are 2-level deep
    if (grp1.basename() == "Epics::EpicsPv") {
      
      for (hdf5pp::Group grp2 = subgiter.next(); grp2.valid(); grp2 = subgiter.next()) {
        hdf5pp::GroupIter subgiter2(grp2);
        for (hdf5pp::Group grp3 = subgiter2.next(); grp3.valid(); grp3 = subgiter2.next()) {
          if (grp3.hasChild("time")) {
            Hdf5DatasetIter begin(grp3);
            Hdf5DatasetIter end(grp3, Hdf5DatasetIter::End);
            m_merger.add(begin, end);
          }
        }
      }

    } else {
      
      for (hdf5pp::Group grp2 = subgiter.next(); grp2.valid(); grp2 = subgiter.next()) {
        if (grp2.hasChild("time")) {
          Hdf5DatasetIter begin(grp2);
          Hdf5DatasetIter end(grp2, Hdf5DatasetIter::End);
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
  std::vector<Hdf5DatasetIterData> data;
  data.reserve(m_merger.size());
  
  if (not m_merger.next(std::back_inserter(data))) {
    return value_type(value_type::Stop);
  } else {
    Hdf5IterData res(value_type::Event);
    for (MergedData::const_iterator it = data.begin(); it != data.end(); ++ it) {
      res.add(it->group, it->index);
    }
    if (not data.empty()) {
      res.setTime(PSTime::Time(data.front().sec, data.front().nsec));
    }
    return res;
  }
}

} // namespace PSHdf5Input
