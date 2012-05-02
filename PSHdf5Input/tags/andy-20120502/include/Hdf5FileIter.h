#ifndef PSHDF5INPUT_HDF5FILEITER_H
#define PSHDF5INPUT_HDF5FILEITER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5FileIter.
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
#include "hdf5pp/File.h"
#include "hdf5pp/Group.h"
#include "PSHdf5Input/Hdf5IterData.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace PSHdf5Input {
class Hdf5ConfigIter;
}

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSHdf5Input {

/**
 *  @ingroup PSHdf5Input
 *  
 *  @brief Iterator class which iterates over events in a single file.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Hdf5FileIter : boost::noncopyable {
public:

  /// Typedef for the data type returned by this iterator
  typedef Hdf5IterData value_type;

  /**
   *  @brief Constructor takes the name of HDF file.
   *  
   *  @throw FileOpenError if file cannot be open 
   */
  explicit Hdf5FileIter (const std::string& fileName) ;

  // Destructor
  ~Hdf5FileIter () ;

  /**
   *  @brief Returns next object
   *  
   *  If there are no more objects left then always return 
   *  object with the type Hdf5IterData::Stop.
   */
  value_type next();

protected:

private:

  std::string m_fileName;
  hdf5pp::File m_file;
  std::list<hdf5pp::Group> m_groups;
  boost::scoped_ptr<Hdf5ConfigIter> m_configIter;
  int m_runNumber;
  
};

} // namespace PSHdf5Input

#endif // PSHDF5INPUT_HDF5FILEITER_H
