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
#include "O2OTranslator/CvtOptions.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

class ConfigObjectStore;

/// @addtogroup O2OTranslator

/**
 *  @ingroup O2OTranslator
 *
 *  @brief Special converter class for Pds::Acqiris::DataDescV1
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
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
  AcqirisDataDescV1Cvt ( const hdf5pp::Group& group,
                         const std::string& typeGroupName,
                         const Pds::Src& src,
                         const ConfigObjectStore& configStore,
                         const CvtOptions& cvtOptions,
                         int schemaVersion ) ;

  // Destructor
  virtual ~AcqirisDataDescV1Cvt () ;

protected:

  /// method called to create all necessary data containers
  virtual void makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src);

  // typed conversion method
  virtual void fillContainers(hdf5pp::Group group,
                              const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src);

  // fill containers for missing data
  virtual void fillMissing(hdf5pp::Group group,
                           const Pds::TypeId& typeId,
                           const O2OXtcSrc& src);

private:

  typedef H5DataTypes::ObjectContainer<H5DataTypes::AcqirisDataDescV1> DataCont ;
  typedef H5DataTypes::ObjectContainer<H5DataTypes::AcqirisTimestampV1> TimestampCont ;
  typedef H5DataTypes::ObjectContainer<int16_t> WaveformCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  boost::shared_ptr<DataCont> m_dataCont ;
  boost::shared_ptr<TimestampCont> m_timestampCont ;
  boost::shared_ptr<WaveformCont> m_waveformCont ;
  size_t n_miss;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_ACQIRISDATADESCV1CVT_H
