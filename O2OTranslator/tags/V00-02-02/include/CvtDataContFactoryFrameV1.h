#ifndef O2OTRANSLATOR_CVTDATACONTFACTORYFRAMEV1_H
#define O2OTRANSLATOR_CVTDATACONTFACTORYFRAMEV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CvtDataContFactoryFrameV1.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "H5DataTypes/ObjectContainer.h"
#include "H5DataTypes/CameraFrameV1.h"
#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/camera/FrameV1.hh"

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
class CvtDataContFactoryFrameV1  {
public:

  // container type
  typedef H5DataTypes::ObjectContainer<H5Type> container_type ;

  // Default constructor
  CvtDataContFactoryFrameV1 ( const std::string& name, hsize_t chunkSize, int deflate )
    : m_name(name)
    , m_chunkSize(chunkSize)
    , m_deflate(deflate)
  {}

  // Destructor
  ~CvtDataContFactoryFrameV1 () {}

  // main method
  container_type* container( hdf5pp::Group group, const Pds::Camera::FrameV1& data ) const
  {
    hdf5pp::Type type = H5DataTypes::CameraFrameV1::imageType ( data ) ;

    hsize_t chunk = std::max( m_chunkSize/type.size(), hsize_t(1) ) ;

    MsgLog( "CvtDataContFactoryFrameV1", debug, "create container " << m_name << " with chunk size " << chunk ) ;
    return new container_type ( m_name, group, type, chunk, m_deflate ) ;
  }

protected:

private:

  // Data members
  std::string m_name ;
  hsize_t m_chunkSize ;
  int m_deflate ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CVTDATACONTFACTORYFRAMEV1_H
