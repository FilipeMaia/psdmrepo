#ifndef PSHDF5INPUT_HDF5ITERDATA_H
#define PSHDF5INPUT_HDF5ITERDATA_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5IterData.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <iosfwd>

//----------------------
// Base Class Headers --
//----------------------
#include <boost/shared_ptr.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "PSEvt/EventId.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSHdf5Input {

/**
 *  @ingroup PSHdf5Input
 *
 *  @brief Class which defines data returned from Hdf5 iterator classes.
 *  
 *  The data object returned by iterators is actually a collection of 
 *  pointers to the objects stored in HDF5. For every object that is 
 *  returned at the next iteration step we return the Group where it 
 *  is stored and an index inside the dataset(s).  In addition to the 
 *  data objects it also contains the type of the "event" which could be 
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Hdf5IterData  {
public:

  /// Description of the single data object in HDF5 file
  struct SingleDataItem {
    SingleDataItem() {}
    SingleDataItem(const hdf5pp::Group& agroup, int64_t aindex)
      : group(agroup), index(aindex) {}
    hdf5pp::Group group;   ///< Group where object resides
    int64_t      index;   ///< Object index in a datasets
  };

  /// Type of the event for the iterator
  enum EventType {
    Configure,       ///< Generated for every new Configure group
    BeginRun,        ///< Generated for every new Run group
    BeginCalibCycle, ///< Generated for every new CalibCycle group
    Event,           ///< Generated for every event inside CalibCycle group
    EndCalibCycle,   ///< Generated after last event in CalibCycle group
    EndRun,          ///< Generated after last CalibCycle group in Run group
    UnConfigure,     ///< Generated after last Run group in Configure group
    Stop             ///< Generated after last Configure group
  };
  
  /// Typedef for single data object
  typedef SingleDataItem value_type;
  /// Typedef for set of data objects
  typedef std::vector<value_type> seq_type;
  /// Typedef for data object iterator
  typedef seq_type::const_iterator const_iterator;
  
  /// Default constructor
  Hdf5IterData () : m_type(Stop), m_data(), m_eid() {}

  /// Constructor takes event type, use add() method to add data objects
  Hdf5IterData (EventType type, const boost::shared_ptr<PSEvt::EventId>& eid)
    : m_type(type), m_data(), m_eid(eid) {}

  /// Add one more data object
  void add(const hdf5pp::Group& group, int64_t index) {
    m_data.push_back(value_type(group, index));
  }
  
  /// Returns event type for current iteration
  EventType type() const { return m_type; }
  
  /// Returns sequence of data objects
  const seq_type& data() const { return m_data; }

  /// Returns eventId for this instance
  const boost::shared_ptr<PSEvt::EventId>& eventId() const { return m_eid; }
  
protected:

private:

  EventType m_type;     ///< Event type for current iteration
  seq_type m_data;      ///< Set of data objects
  boost::shared_ptr<PSEvt::EventId> m_eid;  ///< EventId for this event data
};

/// Standard stream insertion operator for enum type
std::ostream&
operator<<(std::ostream& out, Hdf5IterData::EventType type);

/// Standard stream insertion operator for data type
std::ostream&
operator<<(std::ostream& out, const Hdf5IterData& data);

} // namespace PSHdf5Input

#endif // PSHDF5INPUT_HDF5ITERDATA_H
