#ifndef O2OTRANSLATOR_CVTDATACONTFACTORYTYPED_H
#define O2OTRANSLATOR_CVTDATACONTFACTORYTYPED_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CvtDataContFactoryTyped.
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
class CvtDataContFactoryTyped  {
public:

  // container type
  typedef H5DataTypes::ObjectContainer<H5Type> container_type ;

  // Default constructor
  CvtDataContFactoryTyped ( const std::string& name, hsize_t chunkSize, int deflate, int shuffle = false)
    : m_name(name)
    , m_chunkSize(chunkSize)
    , m_deflate(deflate)
    , m_shuffle(shuffle)
  {
  }

  // Destructor
  ~CvtDataContFactoryTyped () {}

  // main method
  container_type* container( hdf5pp::Group group, hdf5pp::Type type ) const
  {
    MsgLog( "CvtDataContFactoryTyped", debug, "create container " << m_name << " with chunk size " << m_chunkSize ) ;
    return new container_type ( m_name, group, type, m_chunkSize, m_deflate, m_shuffle ) ;
  }

protected:

private:

  // Data members
  std::string m_name ;
  hdf5pp::Type m_type ;
  hsize_t m_chunkSize ;
  int m_deflate ;
  bool m_shuffle;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CVTDATACONTFACTORYTYPED_H
