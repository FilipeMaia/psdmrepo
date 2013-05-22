#ifndef O2OTRANSLATOR_CSPAD2X2CALIBV1CVT_H
#define O2OTRANSLATOR_CSPAD2X2CALIBV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2CalibV1Cvt.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/DataTypeCvtI.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace O2OTranslator {
  class O2OMetaData;
  class CalibObjectStore;
}

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/// @addtogroup O2OTranslator

/**
 *  @ingroup O2OTranslator
 *
 *  Special converter class which reads cspad calibration data from 
 *  external source and prodices HDF5 object from it and also stores 
 *  calibration objects in calibration store.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class CsPad2x2CalibV1Cvt : public DataTypeCvtI {
public:

  // Default constructor
  CsPad2x2CalibV1Cvt(hdf5pp::Group group,
      const std::string& typeGroupName,
      const Pds::Src& src,
      const O2OMetaData& metadata,
      CalibObjectStore& calibStore,
      int schemaVersion);

  // Destructor
  virtual ~CsPad2x2CalibV1Cvt () ;

  /// main method of this class
  virtual void convert ( const void* data, 
                         size_t size,
                         const Pds::TypeId& typeId,
                         const O2OXtcSrc& src,
                         const H5DataTypes::XtcClockTimeStamp& time,
                         Pds::Damage damage ) ;

protected:
  
private:

  // Data members
  std::string m_typeGroupName ;
  const O2OMetaData& m_metadata;
  CalibObjectStore& m_calibStore;
  int m_schemaVersion;
  hdf5pp::Group m_group;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CSPAD2X2CALIBV1CVT_H
