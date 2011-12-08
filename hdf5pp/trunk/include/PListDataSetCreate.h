#ifndef HDF5PP_PLISTDATASETCREATE_H
#define HDF5PP_PLISTDATASETCREATE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PListDataSetCreate.
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

/**
 *  Property list for dataset creation
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace hdf5pp {

class PListDataSetCreate  {
public:

  enum SzipMethod {
	  EntropyCoding = H5_SZIP_EC_OPTION_MASK,
	  NearestNeighbour = H5_SZIP_NN_OPTION_MASK };

  // Default constructor
  PListDataSetCreate () ;

  // Destructor
  ~PListDataSetCreate () ;

  // accessor
  hid_t plist() const { return m_impl.id() ; }

  // set chunk size
  void set_chunk ( int rank, const hsize_t chunk_size[] ) ;

  // set chunk size for rank-1
  void set_chunk ( const hsize_t chunk_size ) ;

  // set deflate compression method
  void set_deflate ( unsigned level ) ;

  // set szip compression method
  void set_szip ( unsigned mask, unsigned block_size ) ;

  // set shuffle "compression"
  void set_shuffle () ;

  // set n-bit compression method
  void set_nbit () ;

protected:

private:

  // Data members
  PListImpl m_impl ;
};

} // namespace hdf5pp

#endif // HDF5PP_PLISTDATASETCREATE_H
