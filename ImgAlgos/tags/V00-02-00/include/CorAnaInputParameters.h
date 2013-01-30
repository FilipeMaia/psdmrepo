#ifndef IMGALGOS_CORANAINPUTPARAMETERS_H
#define IMGALGOS_CORANAINPUTPARAMETERS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CorAnaInputParameters.
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

namespace ImgAlgos {

/// @addtogroup ImgAlgos
/**
 *  @ingroup ImgAlgos
 *  @brief C++ source file code template.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *  @author Mikhail S. Dubrovin
 */

#define NONDEF std::string("non-defined")
#define INPARS CorAnaInputParameters::instance()

class CorAnaInputParameters  {
public:
  virtual ~CorAnaInputParameters () {std::cout << "D-tor: CorAnaInputParameters::~CorAnaInputParameters ()\n";}

  static CorAnaInputParameters* instance();
  void usage(char* name);
  void parse_input_parameters(int argc, char *argv[]);
  void add_file_to_vector(char* name);
  void close_log_file();
  void print();
  void print_input_parameters();
  std::string get_fname_data(){return m_fname_data;}
  std::string get_fname_tau (){return m_fname_tau; }

  std::ostream& get_ostream() {return *p_out;}
  std::vector<std::string>& get_vector_fnames(){return v_files;}

private:

  CorAnaInputParameters () ;                 // Private so that it can not be called

  static CorAnaInputParameters* m_pInstance; // Singleton instance

  std::string m_logfile;
  std::string m_basedir;
  std::string m_fname_data;
  std::string m_fname_tau;
  std::vector<std::string> v_files;

  std::ostream* p_out;                 // Pointer to the output stream (for log-file)
  std::ofstream m_ofs;                 // Output file-stream if the log-name is defined

  // Copy constructor and assignment are disabled by default
  CorAnaInputParameters ( const CorAnaInputParameters& ) ;
  CorAnaInputParameters& operator = ( const CorAnaInputParameters& ) ;
};

} // namespace ImgAlgos

#endif // IMGALGOS_CORANAINPUTPARAMETERS_H
