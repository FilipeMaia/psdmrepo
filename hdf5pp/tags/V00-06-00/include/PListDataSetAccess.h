#ifndef HDF5PP_PLISTDATASETACCESS_H
#define HDF5PP_PLISTDATASETACCESS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PListDataSetAccess.
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
#include "hdf5pp/PListImpl.h"

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
 *  @brief Property list for dataset access
 *
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class PListDataSetAccess  {
public:

  // Default constructor
  PListDataSetAccess () ;

  // Destructor
  ~PListDataSetAccess () ;

  // accessor
  hid_t plist() const { return m_impl.id(); }

  // define chuk cache properties
  void set_chunk_cache(size_t rdcc_nslots, size_t rdcc_nbytes, double rdcc_w0 = 0.75);

protected:

private:

  // Data members
  PListImpl m_impl ;

};

} // namespace hdf5pp

#endif // HDF5PP_PLISTDATASETACCESS_H
