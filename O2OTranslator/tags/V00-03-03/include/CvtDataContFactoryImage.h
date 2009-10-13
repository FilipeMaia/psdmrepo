#ifndef O2OTRANSLATOR_CVTDATACONTFACTORYIMAGE_H
#define O2OTRANSLATOR_CVTDATACONTFACTORYIMAGE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CvtDataContFactoryImage.
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
#include "hdf5pp/Type.h"
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
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

template <typename H5Type>
class CvtDataContFactoryImage  {
public:

  // container type
  typedef H5DataTypes::ObjectContainer<H5Type> container_type ;

  // Default constructor
  CvtDataContFactoryImage ( const std::string& name, hdf5pp::Type type, hsize_t chunkSize, int deflate )
    : m_name(name)
    , m_type(type)
    , m_chunkSize()
    , m_deflate(deflate)
  {
    m_chunkSize = std::max( chunkSize/type.size(), hsize_t(1) ) ;
  }

  // Destructor
  ~CvtDataContFactoryImage () {}

  // main method
  container_type* container( hdf5pp::Group group ) const
  {
    MsgLog( "CvtDataContFactoryTyped", debug, "create container " << m_name << " with chunk size " << m_chunkSize ) ;
    container_type* cont = new container_type ( m_name, group, m_type, m_chunkSize, m_deflate ) ;
    cont->dataset().template createAttr<const char*> ( "CLASS" ).store("IMAGE") ;
    return cont ;
  }

protected:

private:

  // Data members
  std::string m_name ;
  hdf5pp::Type m_type ;
  hsize_t m_chunkSize ;
  int m_deflate ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CVTDATACONTFACTORYIMAGE_H
