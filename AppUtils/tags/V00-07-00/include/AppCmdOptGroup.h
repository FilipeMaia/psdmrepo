#ifndef APPUTILS_APPCMDOPTGROUP_H
#define APPUTILS_APPCMDOPTGROUP_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AppCmdOptGroup.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace AppUtils {
class AppCmdLine;
class AppCmdOptBase;
}

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace AppUtils {

/// @addtogroup AppUtils

/**
 *  @ingroup AppUtils
 *
 *  @brief Class representing option group.
 *
 *  Grouping options in current implementation is only used for display
 *  purposes when producing help/usage information. Having many options
 *  per application may produce help information that can be difficult to
 *  read, splitting full set of option into small number of groups should
 *  help their organization and improve output.
 *
 *  Options defined via specific option classes can be added either to
 *  the parser class (AppCmdLine) directly or to an option group (AppCmdOptGroup).
 *  This can be done either via the addOption() method of the parser or
 *  option group classes or via the constructor of the option classes which
 *  accepts option group as a parameter (AppCmdLine has an AppCmdOptGroup
 *  class as a base so constructors also accept parser instance). Option
 *  group instances have to be added to parser again either via addGroup()
 *  method of the parser or via the group constructor which accepts parser
 *  instance.
 *
 *  Here is an example of creating three options, each one appearing in
 *  a separate option group:
 *  @code
 *  int main(int argc, char** argv)
 *  {
 *    AppCmdLine parser(argv[0]);
 *
 *    // create two additional option groups
 *    AppCmdOptGroup optgrpIn(parser, "Input options");
 *    AppCmdOptGroup optgrpOut(parser, "Output options");
 *
 *    // make an option, add it to parser
 *    AppCmdOptIncr optVerbose(parser, "v,verbose", "Produce more messages", 0);
 *    // make an option, add it to input group
 *    AppCmdOptSize optBufSize(optgrpIn, "b,buf-size", "size", "Input buffer size, def: 1M", 1024*1024);
 *    // make an option, add it to output group
 *    AppCmdOpt<std::string> optOutFile(optgrpOut, "o", "path", "Output file name", "");
 *
 *    parser.parse(argc, argv);
 *    if (parser.helpWanted()) {
 *      parser.usage(std::cout);
 *      return 0;
 *    }
 *  }
 *  @endcode
 *
 *  And here is the output produced by this program running with -h option:
 *  @code
 *  Usage: ./build/x86_64-rhel5-gcc41-opt/AppUtils/apptest1 [options]
 *
 *    General options:
 *      {-h|-?|--help }         print help message
 *      {-v|--verbose } (incr)  Produce more messages
 *
 *    Input options:
 *      {-b|--buf-size} size    Input buffer size, def: 1M
 *
 *    Output options:
 *      {-o           } path    Output file name
 *  @endcode
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class AppCmdOptGroup  {
public:

  /**
   *  @brief Create options group instance.
   *
   *  Created instance of the options group should be added to the parser using
   *  the parser method AppCmdLine::addGroup().
   *
   *  @param[in] groupName   Group name, something like "Input options".
   */
  AppCmdOptGroup(const std::string& groupName);

  /**
   *  @brief Create option group instance
   *
   *  This constructor creates a group and automatically adds it to the parser.
   *
   *  @param[in] parser      Instance of the parser class
   *  @param[in] groupName   Group name, something like "Input options".
   */
  AppCmdOptGroup(AppCmdLine& parser, const std::string& groupName);

  // Destructor
  virtual ~AppCmdOptGroup();

  /**
   *  @brief Add one option to the group.
   *
   *  The option object supplied is not copied, only its address is remembered.
   *  The lifetime of the argument should extend to the parse() method of the parser.
   *
   *  @param[in] option   Option instance to add to the parser.
   */
  virtual void addOption(AppCmdOptBase& option);

protected:

  // The methods below are protected as they do need to be exposed
  // to user, this interface should only be used by AppCmdLine which
  // is declared as friend. Subclasses may use these methods as well
  // for implementing their own functionality or override them.

  /// Type for the list of options
  typedef std::vector<AppCmdOptBase*> OptionsList;

  /**
   *  Get the name of the group.
   */
  virtual const std::string& groupName() const { return m_name; }

  /**
   *  Get the list of defined options.
   */
  virtual const OptionsList& options() const { return m_options; }

private:

  // All private methods are accessible to the parser
  friend class AppCmdLine;

  std::string m_name;
  OptionsList m_options;
  
  // This cass is non-copyable
  AppCmdOptGroup ( const AppCmdOptGroup& ) ;
  AppCmdOptGroup& operator = ( const AppCmdOptGroup& ) ;

};

} // namespace AppUtils

#endif // APPUTILS_APPCMDOPTGROUP_H
