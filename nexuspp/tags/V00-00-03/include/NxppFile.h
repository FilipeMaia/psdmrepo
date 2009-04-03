#ifndef NEXUSPP_NXPPFILE_H
#define NEXUSPP_NXPPFILE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class NxppFile.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "nexus/napi.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "nexuspp/NxppDataSet.h"
#include "nexuspp/NxppExceptions.h"
#include "nexuspp/NxppTypeTraits.h"

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

class NxppFile  {
public:

  enum OpenMode { Read = NXACC_READ,
                  Update = NXACC_RDWR,
                  CreateHdf5 = NXACC_CREATE5,
                  CreateXml = NXACC_CREATEXML } ;

  // factory method
  static NxppFile* open ( const std::string& fileName, OpenMode mode = Read ) ;

  // Destructor
  ~NxppFile () {}

  // Close the file
  void close () {
    if ( NXclose( &m_fileId ) != NX_OK ) {
      throw NxppNexusException ( "NxppFile::close", "NXmakegroup" ) ;
    }
  }

  // create the group, open it
  void createGroup( const std::string& name, const std::string& type ) {
    if ( NXmakegroup( m_fileId, name.c_str(), type.c_str() ) != NX_OK ) {
      throw NxppNexusException ( "NxppFile::createGroup", "NXmakegroup" ) ;
    }
    if ( NXopengroup( m_fileId, name.c_str(), type.c_str() ) != NX_OK ) {
      throw NxppNexusException ( "NxppFile::createGroup", "NXopengroup" ) ;
    }
  }

  // open existing group
  void openGroup( const std::string& name, const std::string& type ) {
    if ( NXopengroup( m_fileId, name.c_str(), type.c_str() ) != NX_OK ) {
      throw NxppNexusException ( "NxppFile::openGroup", "NXopengroup" ) ;
    }
  }

  // close the group
  void closeGroup() {
    if ( NXclosegroup( m_fileId ) != NX_OK ) {
      throw NxppNexusException ( "NxppFile::closeGroup", "NXclosegroup" ) ;
    }
  }

  // make new data set
  template <typename DataType>
  NxppDataSet<DataType>* makeDataSet( const std::string& name, int rank, int dims[] ) ;

  // optimization for rank=1
  template <typename DataType>
  NxppDataSet<DataType>* makeDataSet( const std::string& name, int dims ) ;

  // open existing data set
  template <typename DataType>
  NxppDataSet<DataType>* openDataSet( const std::string& name ) ;

  // add attribute to the currently open object (dataset or group or global)
  template <typename U>
  void addAttribute ( const std::string& name, const U& value ) {
    const void* addr = NxppTypeTraits<U>::dataAddress(value) ;
    int size = NxppTypeTraits<U>::size(value) ;
    int type = NxppTypeTraits<U>::nxtype ;
    if ( NXputattr ( m_fileId, name.c_str(), (void*)addr, size, type ) != NX_OK ) {
      throw NxppNexusException ( "NxppFile::addAttribute", "NXputattr" ) ;
    }
  }

  // add attribute to the currently open object (dataset or group or global)
  template <typename U>
  void addAttribute ( const std::string& name, int size, const U* ptr ) {
    const void* addr = NxppTypeTraits<U>::dataAddress(ptr) ;
    int type = NxppTypeTraits<U>::nxtype ;
    if ( NXputattr ( m_fileId, name.c_str(), (void*)addr, size, type ) != NX_OK ) {
      throw NxppNexusException ( "NxppFile::addAttribute", "NXputattr" ) ;
    }
  }

protected:

  // Default constructor
  NxppFile () : m_fileId() {}
  explicit NxppFile ( NXhandle fileId ) : m_fileId(fileId) {}

private:

  // Data members
  NXhandle m_fileId ;

  // Copy constructor and assignment are disabled by default
  NxppFile ( const NxppFile& ) ;
  NxppFile operator = ( const NxppFile& ) ;

};

// factory method
inline
NxppFile*
NxppFile::open ( const std::string& fileName, OpenMode mode ) {
  NXhandle fileId ;
  if ( NXopen ( fileName.c_str(), NXaccess(mode), &fileId ) != NX_OK ) {
    return 0 ;
  } else {
    return new NxppFile ( fileId ) ;
  }
}

// make new data set
template <typename DataType>
NxppDataSet<DataType>*
NxppFile::makeDataSet( const std::string& name, int rank, int dims[] )
{
  int status = NXmakedata ( m_fileId,
                            name.c_str(),
                            NxppTypeTraits<DataType>::nxtype,
                            rank,
                            dims ) ;
  if ( status != NX_OK ) {
    throw NxppNexusException ( "NxppFile::makeDataSet", "NXmakedata" ) ;
  }
  status = NXopendata ( m_fileId, name.c_str() ) ;
  if ( status != NX_OK ) {
    throw NxppNexusException ( "NxppFile::makeDataSet", "NXopendata" ) ;
  }

  return new NxppDataSet<DataType>( m_fileId );
}

// optimization for rank=1
template <typename DataType>
NxppDataSet<DataType>*
NxppFile::makeDataSet( const std::string& name, int dims )
{
  int status = NXmakedata ( m_fileId,
                            name.c_str(),
                            NxppTypeTraits<DataType>::nxtype,
                            1,
                            &dims ) ;
  if ( status != NX_OK ) {
    throw NxppNexusException ( "NxppFile::makeDataSet", "NXmakedata" ) ;
  }
  status = NXopendata ( m_fileId, name.c_str() ) ;
  if ( status != NX_OK ) {
    throw NxppNexusException ( "NxppFile::makeDataSet", "NXopendata" ) ;
  }

  return new NxppDataSet<DataType>( m_fileId );
}

// open existing data set
template <typename DataType>
NxppDataSet<DataType>*
NxppFile::openDataSet( const std::string& name )
{
  int status = NXopendata ( m_fileId, name.c_str() ) ;
  if ( status != NX_OK ) {
    throw NxppNexusException ( "NxppFile::makeDataSet", "NXopendata" ) ;
  }

  return new NxppDataSet<DataType>( m_fileId );
}

} // namespace nexuspp

#endif // NEXUSPP_NXPPFILE_H
