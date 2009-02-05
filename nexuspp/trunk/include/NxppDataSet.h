#ifndef NEXUSPP_NXPPDATASET_H
#define NEXUSPP_NXPPDATASET_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class NxppDataSet.
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
#include "nexuspp/NxppTypeTraits.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  C++ wrapper for napi.h
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace nexuspp {

template <typename T>
class NxppDataSet  {
public:

  // Constructor
  NxppDataSet ( NXhandle fileId ) : m_fileId(fileId) {}

  // Destructor
  ~NxppDataSet () {
    // on destruction close the data set
    NXclosedata ( m_fileId ) ;
  }

  // put the data
  bool putData ( T* data ) {
    void* addr = NxppTypeTraits<T>::dataAddress(data) ;
    return NXputdata ( m_fileId, addr ) == NX_OK ;
  }
  bool putData ( T data ) {
    const void* addr = NxppTypeTraits<T>::dataAddress(data) ;
    return NXputdata ( m_fileId, (void*)addr ) == NX_OK ;
  }

  // add attribute
  template <typename U>
  bool addAttribute ( const std::string& name, const U& value ) {
    const void* addr = NxppTypeTraits<U>::dataAddress(value) ;
    int size = NxppTypeTraits<U>::size(value) ;
    int type = NxppTypeTraits<U>::nxtype ;
    return NXputattr ( m_fileId, name.c_str(), (void*)addr, size, type ) == NX_OK ;
  }

  // add attribute
  template <typename U>
  bool addAttribute ( const std::string& name, int size, const U* ptr ) {
    const void* addr = NxppTypeTraits<U>::dataAddress(ptr) ;
    int type = NxppTypeTraits<U>::nxtype ;
    return NXputattr ( m_fileId, name.c_str(), (void*)addr, size, type ) == NX_OK ;
  }

protected:

private:

  // Data members
  NXhandle m_fileId;

  // Copy constructor and assignment are disabled by default
  NxppDataSet ( const NxppDataSet& ) ;
  NxppDataSet operator = ( const NxppDataSet& ) ;

};

} // namespace nexuspp

#endif // NEXUSPP_NXPPDATASET_H
