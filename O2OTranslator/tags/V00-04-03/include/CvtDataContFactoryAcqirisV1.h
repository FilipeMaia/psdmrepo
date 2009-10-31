#ifndef O2OTRANSLATOR_CVTDATACONTFACTORYACQIRISV1_H
#define O2OTRANSLATOR_CVTDATACONTFACTORYACQIRISV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CvtDataContFactoryAcqiris.
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
#include "H5DataTypes/AcqirisDataDescV1.h"
#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/acqiris/ConfigV1.hh"

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
class CvtDataContFactoryAcqirisV1  {
public:

  // container type
  typedef H5DataTypes::ObjectContainer<H5Type> container_type ;

  // Default constructor
  CvtDataContFactoryAcqirisV1 ( const std::string& name, hsize_t chunkSize, int deflate, char mode )
    : m_name(name)
    , m_chunkSize(chunkSize)
    , m_deflate(deflate)
    , m_mode(mode)
  {}

  // Destructor
  ~CvtDataContFactoryAcqirisV1 () {}

  // main method
  container_type* container( hdf5pp::Group group, const Pds::Acqiris::ConfigV1& config ) const
  {
    hdf5pp::Type type ;
    switch ( m_mode ) {
      case 'T':
        type = H5DataTypes::AcqirisDataDescV1::timestampType ( config ) ;
        break ;
      case 'W':
        type = H5DataTypes::AcqirisDataDescV1::waveformType ( config ) ;
        break ;
    }

    hsize_t chunk = std::max( m_chunkSize/type.size(), hsize_t(1) ) ;

    MsgLog( "CvtDataContFactoryAcqirisV1", debug, "create container " << m_name << " with chunk size " << chunk ) ;
    return new container_type ( m_name, group, type, chunk, m_deflate ) ;
  }

protected:

private:

  // Data members
  std::string m_name ;
  hsize_t m_chunkSize ;
  int m_deflate ;
  char m_mode ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CVTDATACONTFACTORYACQIRISV1_H
