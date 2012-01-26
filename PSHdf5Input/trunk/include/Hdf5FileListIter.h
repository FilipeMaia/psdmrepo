#ifndef PSHDF5INPUT_HDF5FILELISTITER_H
#define PSHDF5INPUT_HDF5FILELISTITER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5FileListIter.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <list>
#include <boost/utility.hpp>
#include <boost/scoped_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSHdf5Input/Hdf5IterData.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace PSHdf5Input {
class Hdf5FileIter;
}

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSHdf5Input {

/**
 *  @ingroup PSHdf5Input
 *  
 *  @brief Iterator class which merges iterators from several files.
 *  
 *  This is the class which is a merge iterator for several HDF files.
 *  It opens iterator for every file in the list sequentially and 
 *  iterates over each file until end.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see Hdf5FileIter
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Hdf5FileListIter : boost::noncopyable {
public:

  /// Typedef for the data type returned by this iterator
  typedef Hdf5IterData value_type;
  
  /**
   *  @brief  Constructor takes a list of file names.
   *  
   *  The list of files passed to constructor should be 
   *  sorted in the order in which client wants these files
   *  to be scanned.
   */
  explicit Hdf5FileListIter (const std::list<std::string>& fileNames);

  // Destructor
  ~Hdf5FileListIter () ;

  /**
   *  @brief Returns next object
   *  
   *  If there are no more objects left then always return 
   *  object with the type Hdf5IterData::Stop.
   */
  value_type next();
  
protected:

private:

  std::list<std::string> m_fileNames; ///< List of flie names to process
  boost::scoped_ptr<Hdf5FileIter> m_fileIter;  ///< Iterator for currently open file

};

} // namespace PSHdf5Input

#endif // PSHDF5INPUT_HDF5FILELISTITER_H
