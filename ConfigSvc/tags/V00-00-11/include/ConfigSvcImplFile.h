#ifndef CONFIGSVC_CONFIGSVCIMPLFILE_H
#define CONFIGSVC_CONFIGSVCIMPLFILE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigSvcImplFile.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <map>
#include <iosfwd>

//----------------------
// Base Class Headers --
//----------------------
#include "ConfigSvc/ConfigSvcImplI.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ConfigSvc {

/**
 *  Implementation of the configuration service based on INI file format.
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

class ConfigSvcImplFile : public ConfigSvcImplI {
public:

  // Constructor
  ConfigSvcImplFile () ;
  ConfigSvcImplFile (const std::string& file) ;
  ConfigSvcImplFile (std::istream& stream, const std::string& file="<stream>") ;

  // Destructor
  virtual ~ConfigSvcImplFile () ;

  // Get the value of a single parameter, returns zero pointer 
  // if parameter is not there. 
  virtual boost::shared_ptr<const std::string>
    get(const std::string& section, const std::string& param) const;

  // get the value of a single parameter as sequence, returns zero pointer
  // if parameter is not there
  virtual boost::shared_ptr<const std::list<std::string> > 
    getList(const std::string& section, const std::string& param) const;

  // set the value of the parameter, if parameter already exists it will be replaced
  virtual void put(const std::string& section, 
                   const std::string& param, 
                   const std::string& value);

  // get a list of all parameters, or an empty list if the section is not found
  virtual std::list<std::string> getKeys(const std::string& section) const;

protected:

  // read input file from stream
  void readStream(std::istream& in, const std::string& name);
  
private:

  typedef std::map<std::string, boost::shared_ptr<std::string> > ParamMap;
  typedef std::map<std::string, ParamMap> SectionMap;
  typedef std::map<std::string, boost::shared_ptr<std::list<std::string> > > ParamListMap;
  typedef std::map<std::string, ParamListMap> SectionListMap;
  
  // Data members
  SectionMap m_config;
  mutable SectionListMap m_lists;
  
};

} // namespace ConfigSvc

#endif // CONFIGSVC_CONFIGSVCIMPLFILE_H
