//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class InputParameters...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// Headers --
//-----------------------
#include "PSXtcQC/InputParameters.h"
#include <stdlib.h>
#include <unistd.h>  // getopt, getopt_long, getopt_long_only, optarg, optind, opterr, optopt

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSXtcQC {

// Global static pointer used to ensure a single instance of the class.
InputParameters* InputParameters::m_pInstance = NULL; 

//----------------
// Constructors --
//----------------

InputParameters::InputParameters()
{
  std::cout << "!!! Single instance of class InputParameters is created !!!\n";
}

//----------------

InputParameters* InputParameters::instance()
{
  if( !m_pInstance ) m_pInstance = new InputParameters();
  return m_pInstance;
}

//===================

void InputParameters::usage(char* name)
{
  std::cout << "Usage: " << name << " [-h] [-l <logfile>] [-b <basedir>] <fname1> [<fname2> [<fname3> ...]]\n";
}

//===================

void InputParameters::add_file_to_vector(char* name)
{
  std::string fname;
  fname = (m_basedir==NONDEF) ? name : m_basedir + "/" + name;
  v_xfiles.push_back(fname);
  std::cout << "InputParameters::add_file_to_vector: " << fname.c_str() << "\n";
}

//===================

void InputParameters::parse_input_parameters(int argc, char *argv[])
{
  // parse standard option-arguments:
  m_logfile = NONDEF;
  m_basedir = NONDEF;
  int   c;
  while ((c=getopt(argc, argv, ":hl:b:")) != -1) {

    switch (c) {
    case 'l' :
      m_logfile = optarg;
      std::cout << "logfile: " << m_logfile << std::endl;
      break;
    case 'b' :
      m_basedir = optarg;
      std::cout << "basedir: " << m_basedir << std::endl;
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
  if (v_xfiles.size()<1) { 
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

void InputParameters::close_log_file()
{
    std::cout << "InputParameters::close_log_file()\n";
    if (m_logfile != NONDEF)
      m_ofs.close();
}

//----------------

void InputParameters::print()
{
  std::ostream &out = get_ostream();
  out << "InputParameters::print()\n";
}

//----------------

void InputParameters::print_input_parameters()
{
  std::ostream &out = get_ostream();
  out << "InputParameters::print_input_parameters()\n";
  out << "  log file: " << m_logfile << "\n";
  out << "  base dir: " << m_basedir << "\n";
  out << "  \nList of input xtc files:\n";  
  for(std::vector<std::string>::iterator it=v_xfiles.begin(); it!=v_xfiles.end(); it++)
    out << "  " << *it << "\n";
  out << "\n";  
}

//----------------
//----------------


} // namespace PSXtcQC
