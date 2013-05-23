#ifndef PSHDF5INPUT_HDF5CALIBCYCLEITER_H
#define PSHDF5INPUT_HDF5CALIBCYCLEITER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5CalibCycleIter.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <list>
#include <boost/scoped_ptr.hpp>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "PSHdf5Input/Hdf5DatasetIter.h"
#include "PSHdf5Input/Hdf5IterData.h"
#include "PSHdf5Input/MultiMerge.h"

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
 *  @brief Iterator class which iterates over events in a single CalibCycle group.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Hdf5CalibCycleIter : boost::noncopyable {
public:

  /// Typedef for the data type returned by this iterator
  typedef Hdf5IterData value_type;

  // Default constructor
  explicit Hdf5CalibCycleIter (const hdf5pp::Group& grp, int runNumber, unsigned schemaVersion, bool fullTsFormat) ;

  // Destructor
  ~Hdf5CalibCycleIter () ;

  /// get its group
  hdf5pp::Group& group() { return m_grp; }

  /**
   *  @brief Returns next object
   *  
   *  If there are no more objects left then always return 
   *  object with the type Hdf5IterData::Stop.
   */
  value_type next();

protected:

private:

  hdf5pp::Group m_grp;   ///< CalibCycle group
  int m_runNumber;
  unsigned m_schemaVersion;
  bool m_fullTsFormat;
  MultiMerge<Hdf5DatasetIter> m_merger;

};

} // namespace PSHdf5Input

#endif // PSHDF5INPUT_HDF5CALIBCYCLEITER_H
