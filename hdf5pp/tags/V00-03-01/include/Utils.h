#ifndef HDF5PP_UTILS_H
#define HDF5PP_UTILS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Utils.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "hdf5pp/DataSet.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace hdf5pp {

/// @addtogroup hdf5pp

/**
 *  @ingroup hdf5pp
 *
 *  @brief Set of utility methods to facilitate work with HDF5 library
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Utils  {
public:

  /**
   *  @brief Read one object from dataset
   *
   *  Reads one object from scalar or 1-dimensional dataset. If dataset is scalar then
   *  index must be set to -1, any other value means that dataset is 1-dimensional
   *  and an object at given index will be read. Template parameter defines the type
   *  of the returned object, this class must define static method native_type()
   *  which returns hdf5pp::Type object.
   *
   *  @param[in] ds    dataset object
   *  @param[in] index Object index, if negative then dataset must be scalar
   *
   *  @throw hdf5pp::Exception
   */
  template <typename Data>
  static boost::shared_ptr<Data> readDataSet(hdf5pp::DataSet<Data> ds, hsize_t index = -1)
  {
    hdf5pp::DataSpace file_dsp = ds.dataSpace();
    if (index != hsize_t(-1)) file_dsp.select_single(index);
    boost::shared_ptr<Data> ptr = boost::make_shared<Data>();
    ds.read(hdf5pp::DataSpace::makeScalar(), file_dsp, ptr.get(), Data::native_type());
    return ptr;
  }


  /**
   *  @brief Read object from a named dataset.
   *
   *  Meaning and operation are the same as for readDataSet() method, the difference
   *  is that dataset with the given name and parent is open first, then readDataSet()
   *  is called on that object.
   *
   *  @param[in] group   Group object, parent of the dataset.
   *  @param[in] dataset Dataset name
   *  @param[in] index   Object index, if negative then dataset must be scalar
   *
   *  @throw hdf5pp::Exception
   */
  template <typename Data>
  static boost::shared_ptr<Data> readGroup(hdf5pp::Group group, const std::string& dataset, hsize_t index = -1)
  {
    return readDataSet(group.openDataSet<Data>(dataset), index);
  }

};

} // namespace hdf5pp

#endif // HDF5PP_UTILS_H
