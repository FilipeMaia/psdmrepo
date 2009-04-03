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
#include "nexuspp/NxppExceptions.h"
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

  // get info about dataset
  void getInfo ( int* rank, int dims[], int *type=0 ) {
    int atype ;
    if ( NXgetinfo( m_fileId, rank, dims, type ? type : &atype ) != NX_OK ) {
      throw NxppNexusException ( "NxppDataSet::getInfo", "NXgetinfo" ) ;
    }
  }

  // put the data
  void putData ( const T* data ) {
    const void* addr = NxppTypeTraits<T>::dataAddress(data) ;
    if ( NXputdata ( m_fileId, (void*)addr ) != NX_OK ) {
      throw NxppNexusException ( "NxppDataSet::putData", "NXputdata" ) ;
    }
  }
  void putData ( T data ) {
    const void* addr = NxppTypeTraits<T>::dataAddress(data) ;
    if ( NXputdata ( m_fileId, (void*)addr ) != NX_OK ) {
      throw NxppNexusException ( "NxppDataSet::putData", "NXputdata" ) ;
    }
  }

  void putSlab ( const T* data, int start[], int size[] ) {
    const void* addr = NxppTypeTraits<T>::dataAddress(data) ;
    if ( NXputslab ( m_fileId, (void*)addr, start, size ) != NX_OK ) {
      throw NxppNexusException ( "NxppDataSet::putSlab", "NXputslab" ) ;
    }
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
