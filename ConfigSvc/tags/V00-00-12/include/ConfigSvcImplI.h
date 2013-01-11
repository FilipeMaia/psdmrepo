#ifndef CONFIGSVC_CONFIGSVCIMPLI_H
#define CONFIGSVC_CONFIGSVCIMPLI_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigSvcImplI.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <list>
#include <string>
#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------


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
 *  Interface for implementation classes for configuration service.
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

class ConfigSvcImplI : boost::noncopyable {
public:

  // Destructor
  virtual ~ConfigSvcImplI () {}

  // Get the value of a single parameter, returns zero pointer 
  // if parameter is not there. 
  virtual boost::shared_ptr<const std::string>
    get(const std::string& section, const std::string& param) const = 0;

  // get the value of a single parameter as sequence, returns zero pointer
  // if parameter is not there
  virtual boost::shared_ptr<const std::list<std::string> > 
    getList(const std::string& section, const std::string& param) const = 0;

  // set the value of the parameter, if parameter already exists it will be replaced
  virtual void put(const std::string& section, 
                   const std::string& param, 
                   const std::string& value) = 0;

  // get a list of all parameters, or an empty list if the section is not found
  virtual std::list<std::string> getKeys(const std::string& section) const = 0;

protected:

};

} // namespace ConfigSvc

#endif // CONFIGSVC_CONFIGSVCIMPLI_H
