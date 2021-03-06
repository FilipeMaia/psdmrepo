#ifndef O2OTRANSLATOR_CVTDATACONTFACTORYDEF_H
#define O2OTRANSLATOR_CVTDATACONTFACTORYDEF_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CvtDataContFactoryDef.
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
#include "H5DataTypes/ObjectContainer.h"
#include "hdf5pp/Group.h"
#include "MsgLogger/MsgLogger.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
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

template <typename H5Type>
class CvtDataContFactoryDef  {
public:

  // container type
  typedef H5DataTypes::ObjectContainer<H5Type> container_type ;

  // Default constructor
  CvtDataContFactoryDef ( const std::string& name,
                          hsize_t chunkSize,
                          int deflate,
                          hdf5pp::Type type = H5Type::stored_type() )
    : m_name(name)
    , m_chunkSize()
    , m_deflate(deflate)
    , m_type(type)
  {
    m_chunkSize = std::max ( chunkSize / sizeof(H5Type), hsize_t(1) ) ;
  }

  // Destructor
  ~CvtDataContFactoryDef () {}

  // main method
  container_type* container( hdf5pp::Group group ) const
  {
    MsgLog( "CvtDataContFactoryDef", debug, "create container " << m_name << " with chunk size " << m_chunkSize ) ;
    return new container_type ( m_name, group, m_type, m_chunkSize, m_deflate ) ;
  }

protected:

private:

  // Data members
  std::string m_name ;
  hsize_t m_chunkSize ;
  int m_deflate ;
  hdf5pp::Type m_type ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CVTDATACONTFACTORYDEF_H
