#ifndef O2OTRANSLATOR_CVTDATACONTFACTORYEPICS_H
#define O2OTRANSLATOR_CVTDATACONTFACTORYEPICS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CvtDataContFactoryEpics.
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
#include "pdsdata/epics/EpicsPvData.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
 *  Data container factory for EPICS types.
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

class CvtDataContFactoryEpics  {
public:

  // container type
  typedef H5DataTypes::ObjectContainer<Pds::EpicsPvHeader> container_type ;

  // constructor
  CvtDataContFactoryEpics (const std::string& name,  hsize_t chunkSize, int deflate)
    : m_name(name)
    , m_chunkSize(chunkSize)
    , m_deflate(deflate)
  {
  }

  // Destructor
  ~CvtDataContFactoryEpics () {}

  // main method
  container_type* container( hdf5pp::Group group, const Pds::EpicsPvHeader& pv ) const ;

  // get the type for given PV
  static hdf5pp::Type native_type( const Pds::EpicsPvHeader& pv ) { return hdf_type(pv,true) ; }
  static hdf5pp::Type stored_type( const Pds::EpicsPvHeader& pv ) { return hdf_type(pv,false) ; }

protected:

  static hdf5pp::Type hdf_type( const Pds::EpicsPvHeader& pv, bool native ) ;

private:

  // Data members
  std::string m_name ;
  hsize_t m_chunkSize ;
  int m_deflate ;
};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CVTDATACONTFACTORYEPICS_H
