//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FileFinder...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcQC/FileFinder.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

namespace PSXtcQC {

//----------------
// Constructors --
//----------------

FileFinder::FileFinder (const std::string& path) : m_path(path) 
{
  //print();
  //print_dir_content();
  make_list_of_chunks();
  print_list_of_chunks();
}

//----------------

void FileFinder::print()
{
  std::cout << "FileFinder::print(): Print path parts available in boost::filesystem::path\n"; 
  std::cout << "  Input path=" << m_path << "\n";
  if ( m_path == "file-name-is-non-defined" ) return;

  fs::path path = m_path;
  std::cout << "stem          = " << path.stem().string() << "\n";            // e158-r0150-s00-c00
  std::cout << "extension()   = " << path.extension().string() << "\n";       // .xtc
  std::cout << "filename()    = " << path.filename().string() << "\n";        // e158-r0150-s00-c00.xtc
  std::cout << "parent_path() = " << path.parent_path().string() << "\n";     // /reg/d/psdm/CXI/cxi49012/xtc
 }

//----------------

void FileFinder::print_dir_content()
{
  std::cout << "FileFinder::print_dir_content(): Input path=" << m_path << "\n";
  if ( m_path == "file-name-is-non-defined" ) return;

  fs::path path = m_path;
  fs::path dir  = path.parent_path();

  typedef fs::directory_iterator dir_iter;
  for (dir_iter dit = dir_iter( dir ); dit != dir_iter(); ++ dit) {
    std::cout << dit->path().stem().string() << "\n";  
  } 
}

//----------------

void FileFinder::make_list_of_chunks()
{
  std::cout << "FileFinder::make_list_of_chunks(): Input path=" << m_path << "\n";
  if ( m_path == "file-name-is-non-defined" ) return;

  fs::path path = m_path;                          // i.e.: /reg/d/psdm/CXI/cxi49012/xtc/e158-r0150-s00-c01.xtc
  fs::path dir  = path.parent_path();              // i.e.: /reg/d/psdm/CXI/cxi49012/xtc

  static int fname_len = 16;
  std::string stem     = path.stem().string();     // i.e.: e158-r0150-s00-c01
  std::string subname  = stem.substr(0,fname_len); // i.e.: e158-r0150-s00-c
  std::cout << "subname=" << subname << "\n";

  m_list_chunks.clear();

  typedef fs::directory_iterator dir_iter;
  for (dir_iter dit = dir_iter( dir ); dit != dir_iter(); ++ dit) { // loop over files in dir 

    if(dit->path().extension().string() != ".xtc") continue;        // skip non-xtc files

    std::string fstem = dit->path().stem().string();
    if (fstem.substr(0,fname_len) != subname) continue;             // skip files with different name part

    //std::cout << dit->path().string() << "\n";  
    m_list_chunks.push_back(dit->path().string());
  } 
  m_list_chunks.sort();
}

//----------------

void FileFinder::print_list_of_chunks()
{
  std::cout << "FileFinder::print_list_of_chunks()\n";
  for( std::list<std::string>::const_iterator it  = m_list_chunks.begin(); 
                                              it != m_list_chunks.end(); ++it)
    std::cout << *it << "\n";
}
//----------------

std::vector<std::string> FileFinder::get_vect_of_chunks()
{
  m_vect_chunks.clear();
  for( std::list<std::string>::const_iterator it  = m_list_chunks.begin(); 
                                              it != m_list_chunks.end(); ++it)
    m_vect_chunks.push_back(*it);

  return m_vect_chunks;
}

//----------------
} // namespace PSXtcQC
