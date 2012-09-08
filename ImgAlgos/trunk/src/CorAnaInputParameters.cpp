//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CorAnaInputParameters...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// Headers --
//-----------------------
#include "ImgAlgos/CorAnaInputParameters.h"
#include <stdlib.h>

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

// Global static pointer used to ensure a single instance of the class.
CorAnaInputParameters* CorAnaInputParameters::m_pInstance = NULL; 

//----------------
// Constructors --
//----------------

CorAnaInputParameters::CorAnaInputParameters()
{
  std::cout << "!!! Single instance of the class CorAnaInputParameters is created !!!\n";
}

//----------------

CorAnaInputParameters* CorAnaInputParameters::instance()
{
  if( !m_pInstance ) m_pInstance = new CorAnaInputParameters();
  return m_pInstance;
}

//===================

void CorAnaInputParameters::usage(char* name)
{
  std::cout << "Usage: " << name << " [-h] [-l <logfile>] [-b <basedir>] [-f <fname>] [<fname1> [<fname2> [<fname3> ...]]]\n";
}

//===================

void CorAnaInputParameters::add_file_to_vector(char* name)
{
  std::string fname;
  fname = (m_basedir==NONDEF) ? name : m_basedir + "/" + name;
  v_files.push_back(fname);
  std::cout << "CorAnaInputParameters::add_file_to_vector: " << fname.c_str() << "\n";
}

//===================

void CorAnaInputParameters::parse_input_parameters(int argc, char *argv[])
{
  // parse standard option-arguments:
  m_logfile = NONDEF;
  m_basedir = NONDEF;
  int   c;
  while ((c=getopt(argc, argv, ":hl:b:f:")) != -1) {

    switch (c) {
    case 'l' :
      m_logfile = optarg;
      std::cout << "logfile: " << m_logfile << std::endl;
      break;
    case 'b' :
      m_basedir = optarg;
      std::cout << "basedir: " << m_basedir << std::endl;
      break;
    case 'f' :
      add_file_to_vector(optarg);
      break;
    case 'h' :
      usage(argv[0]);
      break;
    case ':' :
      std::cout << "Missing argument\n";          
      usage(argv[0]);
      exit(0);
    case '?' :
      std::cout << "Non-defined option: -" << char(optopt) <<"\n";          
      usage(argv[0]);
      exit(0);
    default:
      std::cout << "Default should never happen...";
      abort();
    }
  }

  // parse non-optional arguments:
  // We assume, that all xtc file names are listed as non-optional arguments.
  for (int index = optind; index < argc; index++) {
      std::cout << "Non-option argument: " <<  argv[index] << "\n";
      add_file_to_vector(argv[index]);
  }

  // if the file-name still is not defined 
  if (v_files.size()<1) { 
      std::cout << "File name(s) is not defined... At least one file name is required.\n";          
      usage(argv[0]); 
      exit(0);
  }

  // Select the output log-file stream  
  if (m_logfile == NONDEF) {
    std::cout << "Output will be directed to stdout\n";
    p_out = &std::cout;
  }
  else { 
    std::cout << "Output will be directed in file " << m_logfile << "\n"; 
    m_ofs.open(m_logfile.c_str());
    p_out = &m_ofs;
  }

  //p_out = (m_logfile == NONDEF) ? &std::cout : &m_ofs;

  print_input_parameters();

}

//----------------

void CorAnaInputParameters::close_log_file()
{
  if (m_logfile != NONDEF) {
      m_ofs.close();
      std::cout << "CorAnaInputParameters::close_log_file() The log file " << m_logfile << " is closed.\n";
  }
}

//----------------

void CorAnaInputParameters::print()
{
  std::ostream &out = INPARS -> get_ostream();
  out << "CorAnaInputParameters::print()\n";
}

//----------------

void CorAnaInputParameters::print_input_parameters()
{
  std::ostream &out = INPARS -> get_ostream();
  out << "CorAnaInputParameters::print_input_parameters()\n";
  out << "  Log file: " << m_logfile << "\n";
  out << "  Base dir: " << m_basedir << "\n";
  out << "  List of input files:\n";  
  for(std::vector<std::string>::iterator it=v_files.begin(); it!=v_files.end(); it++)
    out << "    " << *it << "\n";
  out << "\n";  
}

//----------------
//----------------

} // namespace ImgAlgos
