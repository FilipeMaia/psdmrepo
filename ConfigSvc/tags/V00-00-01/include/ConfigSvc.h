#ifndef CONFIGSVC_CONFIGSVC_H
#define CONFIGSVC_CONFIGSVC_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigSvc.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <list>
#include <memory>
#include <string>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ConfigSvc/ConfigSvcImplI.h"
#include "ConfigSvc/ConfigSvcTypeTraits.h"
#include "ConfigSvc/Exceptions.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ConfigSvc {

/**
 *  Configuration service class.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class ConfigSvc  {
public:
  
  // ========================================================================

  // helper classes to handle conversion
  class Result {
  public:
    Result(const boost::shared_ptr<const std::string>& pstr) : m_pstr(pstr) {}
    
    // conversion to final value
    template <typename T>
    operator T() const { return ConfigSvcTypeTraits<T>::fromString(*m_pstr); }
    
  private:
    boost::shared_ptr<const std::string> m_pstr;
  };

  template <typename Def>
  class ResultDef {
  public:
    ResultDef(const boost::shared_ptr<const std::string>& pstr, const Def& def) 
      : m_pstr(pstr), m_def(def) {}
    
    // conversion to final value
    template <typename T>
    operator T() const {
      if ( m_pstr.get() ) {
        return ConfigSvcTypeTraits<T>::fromString(*m_pstr);
      } else {
        return m_def;
      }
    }
    
  private:
    boost::shared_ptr<const std::string> m_pstr;
    Def m_def;
  };
  
  class ResultList {
  public:
    ResultList(const boost::shared_ptr<const std::list<std::string> >& pstr) 
      : m_pstr(pstr) {}
    
    // conversion to final value
    template <typename Container>
    operator Container() const {
      Container res ;
      for (std::list<std::string>::const_iterator it = m_pstr->begin() ; it != m_pstr->end() ; ++ it ) {      
        res.push_back( ConfigSvcTypeTraits<typename Container::value_type>::fromString(*it) );
      }
      return res;
    }
    
  private:
    boost::shared_ptr<const std::list<std::string> > m_pstr;
  };

  template <typename Def>
  class ResultListDef {
  public:
    ResultListDef(const boost::shared_ptr<const std::list<std::string> >& pstr, const Def& def) 
      : m_pstr(pstr), m_def(def) {}
    
    // conversion to final value
    template <typename Container>
    operator Container() const {
      if (not m_pstr.get()) return m_def;
      
      Container res ;
      for (std::list<std::string>::const_iterator it = m_pstr->begin() ; it != m_pstr->end() ; ++ it ) {      
        res.push_back( ConfigSvcTypeTraits<typename Container::value_type>::fromString(*it) );
      }
      return res;
    }
    
  private:
    boost::shared_ptr<const std::list<std::string> > m_pstr;
    Def m_def;
  };

  
  // ========================================================================
  
  
  // get the value of a single parameter, will throw ExceptionMissing if parameter is not there
  Result get(const std::string& section, const std::string& param) const {
    boost::shared_ptr<const std::string> pstr = impl().get(section, param);
    if (not pstr.get()) throw ExceptionMissing(section, param);
    return Result( pstr );
  }
  
  // get the value of a single parameter, use default if not there
  template <typename T>
  ResultDef<T> get(const std::string& section, const std::string& param, const T& def) const
  {
    boost::shared_ptr<const std::string> pstr = impl().get(section, param);
    return ResultDef<T>( pstr, def );
  }
  
  // get the value of a single parameter as sequence, will throw if parameter is not there
  ResultList getList(const std::string& section, const std::string& param) const {
    boost::shared_ptr<const std::list<std::string> > pstr = impl().getList(section, param);
    if (not pstr.get()) throw ExceptionMissing(section, param);
    return ResultList(pstr);
  }
  
  // get the value of a single parameter as sequence, or return default value 
  template <typename Cont>
  ResultListDef<Cont> getList(const std::string& section, const std::string& param, const Cont& def) const
  {
    boost::shared_ptr<const std::list<std::string> > pstr = impl().getList(section, param);
    return ResultListDef<Cont>(pstr, def);
  }
  
protected:

private:

//------------------
// Static Members --
//------------------

public:

  // initialize the service, throws if it has been initialized already
  static void init(std::auto_ptr<ConfigSvcImplI> impl);

private:

  // get the implementation instance, throws if not initialized
  static ConfigSvcImplI& impl();
  
};

} // namespace ConfigSvc

#endif // CONFIGSVC_CONFIGSVC_H
