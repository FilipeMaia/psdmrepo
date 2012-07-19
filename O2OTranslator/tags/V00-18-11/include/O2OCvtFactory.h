#ifndef O2OTRANSLATOR_O2OCVTFACTORY_H
#define O2OTRANSLATOR_O2OCVTFACTORY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OCvtFactory.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <map>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "O2OTranslator/DataTypeCvtI.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace O2OTranslator {
class CalibObjectStore;
class ConfigObjectStore;
class O2OMetaData;
}


//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/// @addtogroup O2OTranslator

/**
 *  @ingroup O2OTranslator
 *
 *  @brief Utility class which instantiates converters for all known data types.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class O2OCvtFactory  {
public:

  typedef boost::shared_ptr<DataTypeCvtI> DataTypeCvtPtr ;
  typedef std::multimap<uint32_t, DataTypeCvtPtr> CvtMap ;

  /**
   *  @brief Method which instantiates converters for all known data types
   */
  static void makeConverters(CvtMap& cvtMap, ConfigObjectStore& configStore, CalibObjectStore& calibStore,
      const O2OMetaData& metadata, int compression);

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OCVTFACTORY_H
