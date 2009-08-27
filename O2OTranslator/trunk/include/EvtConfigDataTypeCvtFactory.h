#ifndef O2OTRANSLATOR_EVTCONFIGDATATYPECVTFACTORY_H
#define O2OTRANSLATOR_EVTCONFIGDATATYPECVTFACTORY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvtConfigDataTypeCvtFactory.
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
#include "MsgLogger/MsgLogger.h"

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
class EvtConfigDataTypeCvtFactory : public DataTypeCvtFactoryI {
public:

  // Default constructor
  EvtConfigDataTypeCvtFactory ( const std::string& grpName,
                          hsize_t chunk_size = 128*1024,
                          int deflate = 1 )
    : m_group()
    , m_grpName(grpName)
    , m_chunk_size(chunk_size)
    , m_deflate(deflate)
  {
  }

  // Destructor
  virtual ~EvtConfigDataTypeCvtFactory () { destroyConverters() ; }

  // Get the converter for given parameter set
  virtual DataTypeCvtI* converter(const Pds::DetInfo& info)
  {
    typename CvtMap::iterator i = m_cvtMap.find( info ) ;
    if ( i != m_cvtMap.end() ) return i->second ;

    // define separate group
    hdf5pp::Group grp ;
    if ( m_group.valid() ) {
      const std::string& grpName = cvtGroupName( m_grpName, info ) ;
      MsgLogRoot( debug, "Creating group " << grpName ) ;
      grp = m_group.createGroup( grpName );
    }

    Converter* cvt = new Converter ( grp, m_chunk_size, m_deflate ) ;
    m_cvtMap.insert ( typename CvtMap::value_type( info, cvt) ) ;

    return cvt ;
  }

  // this method is called at configure transition
  virtual void configure ( const hdf5pp::Group& cfgGroup ) { }

  // this method is called at unconfigure transition
  virtual void unconfigure () { }

  // this method is called at begin-run transition
  virtual void beginRun ( const hdf5pp::Group& runGroup ) {
    m_group = runGroup ;
    for ( typename CvtMap::iterator i = m_cvtMap.begin() ; i != m_cvtMap.end() ; ++ i ) {
      const std::string& grpName = cvtGroupName( m_grpName, i->first ) ;
      MsgLogRoot( debug, "Creating group " << grpName ) ;
      hdf5pp::Group grp = m_group.createGroup( grpName );
      i->second->setGroup ( grp ) ;
    }
  }

  // this method is called at end-run transition
  virtual void endRun () {
    destroyConverters() ;
    m_group = hdf5pp::Group() ;
  }

protected:

  void destroyConverters() {
    // destroy converters
    for ( typename CvtMap::iterator i = m_cvtMap.begin() ; i != m_cvtMap.end() ; ++ i ) {
      delete i->second ;
    }
    m_cvtMap.clear() ;
  }

private:


  typedef std::map<Pds::DetInfo,Converter*,DataTypeCvtFactoryI::CmpDetInfo> CvtMap ;

  // Data members
  hdf5pp::Group m_group ;
  std::string m_grpName ;
  hsize_t m_chunk_size ;
  int m_deflate ;
  CvtMap m_cvtMap ;

  // Copy constructor and assignment are disabled by default
  EvtConfigDataTypeCvtFactory ( const EvtConfigDataTypeCvtFactory& ) ;
  EvtConfigDataTypeCvtFactory operator = ( const EvtConfigDataTypeCvtFactory& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_EVTCONFIGDATATYPECVTFACTORY_H
