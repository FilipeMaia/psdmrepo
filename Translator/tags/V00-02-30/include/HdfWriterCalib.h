#ifndef TRANSLATOR_HDFWRITERCALIB_H
#define TRANSLATOR_HDFWRITERCALIB_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Interface to translation of psana CalibStore data
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <map>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "boost/shared_ptr.hpp"
#include "PSEvt/TypeInfoUtils.h"
#include "Translator/HdfWriterNew.h"

namespace Translator {

  /**
   * @ingroup Translator
   * 
   * @brief returns a list of HdfWriterNew objects for the calib types that the Translator can write.
   * 
   * The list of HdfWriterNew objects can be wrapped in HdfWriterNewDataFromEvent classes to get
   * writers for data in the calibStore.
   *
   * @see HdfWriterNewDataFromEvent, HdfWriterNew
   */
void getHdfWritersForCalibStore(std::vector< boost::shared_ptr<HdfWriterNew> > & calibStoreWriters);

/// typedef for getType2CalibTypesMap
typedef std::map<const std::type_info *, std::vector<const std::type_info *> , 
                                 PSEvt::TypeInfoUtils::lessTypeInfoPtr >  Type2CalibTypesMap;

  /**
   * @ingroup Translator
   * 
   * @brief returns a map identifying what calib data to write for what types.
   * 
   * Different calibration modules may store different kinds of data in the calibStore.
   * Any such data that is useful to Translate should be encoded in map returned by
   * getType2CalibTypesMap. The keys are psana types that can be calibrated. The
   * values are lists of types that may show up in the calibStore (such as pedstals,
   * pixel status, etc). 
   */
void getType2CalibTypesMap(Type2CalibTypesMap & type2calibTypeMap);

} // namesapce Translator

#endif
