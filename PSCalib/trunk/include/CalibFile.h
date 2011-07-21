#ifndef O2OTRANSLATOR_CSPADCALIBV1CVT_H
#define O2OTRANSLATOR_CSPADCALIBV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: CsPadCalibV1Cvt.h 1456 2011-01-27 23:56:59Z salnikov $
//
// Description:
//	Class CsPadCalibV1Cvt.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <stack>
#include <boost/shared_ptr.hpp>

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
namespace pdscalibdata {
  class CsPadPedestalsV1;
}

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
 *  Special converter class which reads cspad calibration data from 
 *  external source and prodices HDF5 object from it and also stores 
 *  calibration objects in calibration store.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id: CsPadCalibV1Cvt.h 1456 2011-01-27 23:56:59Z salnikov $
 *
 *  @author Andrei Salnikov
 */

class CsPadCalibV1Cvt : public DataTypeCvtI {
public:

  // Default constructor
  CsPadCalibV1Cvt ( const std::string& typeGroupName,
          const O2OMetaData& metadata,
          CalibObjectStore& calibStore);

  // Destructor
  virtual ~CsPadCalibV1Cvt () ;

  /// main method of this class
  virtual void convert ( const void* data, 
                         size_t size,
                         const Pds::TypeId& typeId,
                         const O2OXtcSrc& src,
                         const H5DataTypes::XtcClockTime& time ) ;

  /// method called when the driver makes a new group in the file
  virtual void openGroup( hdf5pp::Group group ) ;

  /// method called when the driver closes a group in the file
  virtual void closeGroup( hdf5pp::Group group ) ;

protected:
  
  // find pedestals data
  std::string findCalibFile(const O2OXtcSrc& src, const std::string& datatype) const;
  
private:

  // Data members
  std::string m_typeGroupName ;
  const O2OMetaData& m_metadata;
  CalibObjectStore& m_calibStore;
  std::stack<hdf5pp::Group> m_groups ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CSPADCALIBV1CVT_H
