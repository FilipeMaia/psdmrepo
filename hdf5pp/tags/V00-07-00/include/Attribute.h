#ifndef HDF5PP_ATTRIBUTE_H
#define HDF5PP_ATTRIBUTE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Attribute.
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
#include "hdf5pp/DataSpace.h"
#include "hdf5pp/Exceptions.h"
#include "hdf5pp/TypeTraits.h"

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
 *  Class representing HDF5 attribute.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

template <typename T>
class Attribute  {
public:

  // create new attribute
  static Attribute createAttr ( hid_t parent, const std::string& name, const DataSpace& dspc = DataSpace::makeScalar() ) ;

  // Open existing attribute, returns non-valid object if does not exist
  static Attribute openAttr ( hid_t parent, const std::string& name ) ;

  // Destructor
  ~Attribute () {}

  /// get attribute data space
  const DataSpace& dataSpace() const { return m_dspc ; }

  /// store attribute value (for scalar attributes)
  void store( const T& value ) ;

  /// store attribute value (for arbitrary attributes)
  void store( unsigned size, const T value[] ) ;

  /// read attribute value (for scalar attributes)
  T read() ;

  // returns true if there is a real object behind
  bool valid() const { return m_id.get() ; }

protected:

  // Constructor
  Attribute () {}
  Attribute ( hid_t id, const DataSpace& dspc ) ;

private:

  // deleter for  boost smart pointer
  struct AttrPtrDeleter {
    void operator()( hid_t* id ) {
      if ( id ) H5Aclose ( *id );
      delete id ;
    }
  };

  // Data members
  DataSpace m_dspc;
  boost::shared_ptr<hid_t> m_id ;

};

// Constructor
template <typename T>
Attribute<T>::Attribute ( hid_t id, const DataSpace& dspc )
  : m_dspc( dspc )
  , m_id( new hid_t(id), AttrPtrDeleter() )
{
}

// factory methods
template <typename T>
Attribute<T>
Attribute<T>::createAttr ( hid_t parent, const std::string& name, const DataSpace& dspc )
{
  hid_t aid = H5Acreate2 ( parent, name.c_str(), TypeTraits<T>::stored_type().id(), dspc.id(), H5P_DEFAULT, H5P_DEFAULT ) ;
  if ( aid < 0 ) throw Hdf5CallException( ERR_LOC, "H5Acreate2" ) ;
  return Attribute<T>( aid, dspc ) ;
}

template <typename T>
Attribute<T>
Attribute<T>::openAttr ( hid_t parent, const std::string& name )
{
  htri_t rc = H5Aexists(parent, name.c_str()) ;
  if ( rc < 0 ) throw Hdf5CallException( ERR_LOC, "H5Aexists" ) ;
  if ( rc == 0 ) return Attribute<T>();
  hid_t aid = H5Aopen ( parent, name.c_str(), H5P_DEFAULT ) ;
  if ( aid < 0 ) throw Hdf5CallException( ERR_LOC, "H5Aopen" ) ;
  hid_t dspc = H5Aget_space( aid ) ;
  if ( dspc < 0 ) throw Hdf5CallException( ERR_LOC, "H5Aget_space" ) ;
  return Attribute<T>( aid, DataSpace(dspc) ) ;
}

/// store attribute value (for scalar attributes)
template <typename T>
void
Attribute<T>::store( const T& value )
{
  if ( m_dspc.size() != 1 ) throw Hdf5DataSpaceSizeException ( ERR_LOC );
  herr_t stat = H5Awrite ( *m_id, TypeTraits<T>::native_type().id(), TypeTraits<T>::address(value) ) ;
  if ( stat < 0 ) throw Hdf5CallException( ERR_LOC, "H5Awrite" ) ;
}

/// store attribute value (for arbitrary attributes)
template <typename T>
void
Attribute<T>::store( unsigned size, const T value[] )
{
  if ( m_dspc.size() != size ) throw Hdf5DataSpaceSizeException ( ERR_LOC );
  herr_t stat = H5Awrite ( *m_id, TypeTraits<T>::native_type().id(), (void*)(value) ) ;
  if ( stat < 0 ) throw Hdf5CallException( ERR_LOC, "H5Awrite" ) ;
}

/// read attribute value (for scalar attributes)
template <typename T>
T 
Attribute<T>::read()
{
  if ( m_dspc.size() != 1 ) throw Hdf5DataSpaceSizeException ( ERR_LOC );
  T value;
  herr_t stat = H5Aread ( *m_id, TypeTraits<T>::native_type().id(), TypeTraits<T>::address(value) ) ;
  if ( stat < 0 ) throw Hdf5CallException( ERR_LOC, "H5Aread" ) ;
  return value;
}

} // namespace hdf5pp

#endif // HDF5PP_ATTRIBUTE_H
