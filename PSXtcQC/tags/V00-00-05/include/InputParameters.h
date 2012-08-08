#ifndef PSXTCQC_INPUTPARAMETERS_H
#define PSXTCQC_INPUTPARAMETERS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class InputParameters.
//
//      SINGLETON for input parameters 
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <fstream>  // for ostream, ofstream
#include <iostream> // for cout, puts etc.

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSXtcQC {

/// @addtogroup PSXtcQC
/**
 *  @ingroup PSXtcQC
 *  @brief C++ source file code template.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *  @author Mikhail S. Dubrovin
 */

#define NONDEF std::string("non-defined")
#define INPARS InputParameters::instance()

class InputParameters  {
public:
  virtual ~InputParameters () {std::cout << "D-tor: InputParameters::~InputParameters ()\n";}

  static InputParameters* instance();
  void usage(char* name);
  void parse_input_parameters(int argc, char *argv[]);
  void add_file_to_vector(char* name);
  void close_log_file();
  void print();
  void print_input_parameters();

  std::ostream& get_ostream(){return *p_out;}
  std::vector<std::string>& get_vector_fnames(){return v_xfiles;}

private:

  InputParameters () ;                 // Private so that it can not be called

  static InputParameters* m_pInstance; // Singleton instance

  std::string m_logfile;
  std::string m_basedir;
  std::vector<std::string> v_xfiles;

  std::ostream* p_out;                 // Pointer to the output stream (for log-file)
  std::ofstream m_ofs;                 // Output file-stream if the log-name is defined

  // Copy constructor and assignment are disabled by default
  InputParameters ( const InputParameters& ) ;
  InputParameters& operator = ( const InputParameters& ) ;
};

} // namespace PSXtcQC

#endif // PSXTCQC_INPUTPARAMETERS_H
