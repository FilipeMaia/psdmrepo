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
 *  @brief Class which instantiates converters for all known data types.
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
  typedef CvtMap::iterator iterator ;
  typedef CvtMap::const_iterator const_iterator ;

  /**
   *  @brief Constructor instantiates converters for all known data types
   */
  O2OCvtFactory(ConfigObjectStore& configStore, CalibObjectStore& calibStore,
      const O2OMetaData& metadata, int compression);

  iterator begin() { return m_cvtMap.begin(); }
  iterator end() { return m_cvtMap.end(); }
  const_iterator begin() const { return m_cvtMap.begin(); }
  const_iterator end() const { return m_cvtMap.end(); }

  iterator find(const Pds::TypeId& typeId) { return m_cvtMap.find(typeId.value()); }
  const_iterator find(const Pds::TypeId& typeId) const { return m_cvtMap.find(typeId.value()); }

private:
  
  CvtMap m_cvtMap ;
  
};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OCVTFACTORY_H
