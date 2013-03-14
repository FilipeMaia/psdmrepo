#ifndef PSHDF5INPUT_HDF5CONFIGITER_H
#define PSHDF5INPUT_HDF5CONFIGITER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5ConfigIter.
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
#include "PSHdf5Input/Hdf5IterData.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace PSHdf5Input {
class Hdf5RunIter;
}

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSHdf5Input {

/**
 *  @ingroup PSHdf5Input
 *  
 *  @brief Iterator class which iterates over events in a single Configure group.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Hdf5ConfigIter : boost::noncopyable {
public:

  /// Typedef for the data type returned by this iterator
  typedef Hdf5IterData value_type;

  // Default constructor
  explicit Hdf5ConfigIter (const hdf5pp::Group& grp, int runNumber) ;

  // Destructor
  ~Hdf5ConfigIter () ;

  /**
   *  @brief Returns next object
   *  
   *  If there are no more objects left then always return 
   *  object with the type Hdf5IterData::Stop.
   */
  value_type next();

  /// get its group
  hdf5pp::Group& group() { return m_grp; }

protected:

private:

  hdf5pp::Group m_grp;    ///< Configure group
  int m_runNumber;
  std::list<hdf5pp::Group> m_groups;   ///< List of Run groups
  boost::scoped_ptr<Hdf5RunIter> m_runIter;  ///< Iterator for current run group
  
};

} // namespace PSHdf5Input

#endif // PSHDF5INPUT_HDF5CONFIGITER_H
