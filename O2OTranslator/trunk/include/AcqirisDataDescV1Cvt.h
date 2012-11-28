#ifndef O2OTRANSLATOR_ACQIRISDATADESCV1CVT_H
#define O2OTRANSLATOR_ACQIRISDATADESCV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisDataDescV1Cvt.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/EvtDataTypeCvt.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/AcqirisDataDescV1.h"
#include "pdsdata/acqiris/DataDescV1.hh"
#include "O2OTranslator/CvtDataContainer.h"
#include "O2OTranslator/CvtDataContFactoryDef.h"
#include "O2OTranslator/CvtDataContFactoryTyped.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

class ConfigObjectStore;

/**
 *  Special converter class for Pds::Acqiris::DataDescV1
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

class AcqirisDataDescV1Cvt : public EvtDataTypeCvt<Pds::Acqiris::DataDescV1> {
public:

  typedef Pds::Acqiris::DataDescV1 XtcType ;
  typedef H5DataTypes::AcqirisDataDescV1 H5Type ;

  // Default constructor
  AcqirisDataDescV1Cvt ( const std::string& typeGroupName,
                         const ConfigObjectStore& configStore,
                         hsize_t chunk_size,
                         int deflate ) ;

  // Destructor
  virtual ~AcqirisDataDescV1Cvt () ;

protected:

  /// method called to create all necessary data containers
  virtual void makeContainers(hsize_t chunk_size, int deflate,
      const Pds::TypeId& typeId, const O2OXtcSrc& src);

  // typed conversion method
  virtual void fillContainers(hdf5pp::Group group,
                              const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src);

  /// method called when the driver closes a group in the file
  virtual void closeContainers(hdf5pp::Group group);

private:

  typedef CvtDataContainer<CvtDataContFactoryTyped<uint64_t> > TimestampCont ;
  typedef CvtDataContainer<CvtDataContFactoryTyped<int16_t> > WaveformCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  TimestampCont* m_timestampCont ;
  WaveformCont* m_waveformCont ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_ACQIRISDATADESCV1CVT_H
