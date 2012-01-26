#ifndef PSHDF5INPUT_HDF5DATASETITERDATA_H
#define PSHDF5INPUT_HDF5DATASETITERDATA_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5DatasetIterData.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSHdf5Input {

/**
 *  @brief class for data returned by Hdf5DatasetIter iterator class.
 *  
 *  The data of this type will be merged by the MultiMerge class
 *  so it has to be LessThanComparable. Comparison is based on the
 *  event time.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

struct Hdf5DatasetIterData  {
public:

  /// Compare two objects, time only
  bool operator<(const Hdf5DatasetIterData& other) const 
  {
    if (sec < other.sec) return true;
    if (sec > other.sec) return false;
    if (nsec < other.nsec) return true;
    return false;
  }
  
  hdf5pp::Group group;   ///< Group where objects live
  uint64_t      index;   ///< Object index in a datasets
  uint32_t      sec;     ///< Time (seconds part) at current index
  uint32_t      nsec;    ///< Time (nanoseconds part) at current index
};

} // namespace PSHdf5Input

#endif // PSHDF5INPUT_HDF5DATASETITERDATA_H
