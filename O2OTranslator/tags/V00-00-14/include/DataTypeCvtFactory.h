#ifndef O2OTRANSLATOR_DATATYPECVTFACTORY_H
#define O2OTRANSLATOR_DATATYPECVTFACTORY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataTypeCvtFactory.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <map>
#include <string>

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/DataTypeCvtFactoryI.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/DetInfo.hh"
#include "hdf5pp/Group.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Factory class for converters
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

namespace O2OTranslator {

template <typename Converter>
class DataTypeCvtFactory : public DataTypeCvtFactoryI {
public:

  // Default constructor
  DataTypeCvtFactory ( const std::string& grpName )
    : m_group()
    , m_grpName(grpName)
  {
  }

  // Destructor
  virtual ~DataTypeCvtFactory () { destroyConverters() ; }

  // Get the converter for given parameter set
  virtual DataTypeCvtI* converter(const Pds::DetInfo& info)
  {
    CvtMap::iterator i = m_cvtMap.find( info ) ;
    if ( i != m_cvtMap.end() ) return i->second ;

    // build the name for the group where the object will live
    std::ostringstream str ;
    str << m_grpName << '/' << Pds::DetInfo::name(info.detector()) << '.' << info.detId()
        << ':' << Pds::DetInfo::name(info.device()) << '.' << info.devId() ;
    const std::string& grpName = str.str() ;

    // define separate group
    hdf5pp::Group grp = m_group.createGroup( grpName );

    DataTypeCvtI* cvt = new Converter ( grp ) ;
    m_cvtMap.insert ( CvtMap::value_type( info, cvt ) ) ;

    return cvt ;
  }

  // this method is called at configure transition
  virtual void configure ( const hdf5pp::Group& cfgGroup ) { m_group = cfgGroup ; }

  // this method is called at unconfigure transition
  virtual void unconfigure () {
    destroyConverters() ;
    m_group = hdf5pp::Group() ;
  }

  // this method is called at begin-run transition
  virtual void beginRun ( const hdf5pp::Group& runGroup ) { }

  // this method is called at end-run transition
  virtual void endRun () { }

protected:

  void destroyConverters() {
    // destroy converters
    for ( CvtMap::iterator i = m_cvtMap.begin() ; i != m_cvtMap.end() ; ++ i ) {
      delete i->second ;
    }
    m_cvtMap.clear() ;
  }

private:


  typedef std::map<Pds::DetInfo,DataTypeCvtI*,DataTypeCvtFactoryI::CmpDetInfo> CvtMap ;

  // Data members
  hdf5pp::Group m_group ;
  std::string m_grpName ;
  CvtMap m_cvtMap ;

  // Copy constructor and assignment are disabled by default
  DataTypeCvtFactory ( const DataTypeCvtFactory& ) ;
  DataTypeCvtFactory operator = ( const DataTypeCvtFactory& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_DATATYPECVTFACTORY_H
