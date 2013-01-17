#ifndef PSXTCQC_FILEFINDER_H
#define PSXTCQC_FILEFINDER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FileFinder.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include <list>
#include <vector>
#include <string>
#include <iostream> // for cout, etc.

namespace PSXtcQC {

/// @addtogroup PSXtcQC

/**
 *  @ingroup PSXtcQC
 *
 *  @brief C++ source file code template.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class FileFinder  {
public:

  FileFinder (const std::string& path) ;
  virtual ~FileFinder () {}

  void print();
  void print_dir_content();
  void make_list_of_chunks();
  void print_list_of_chunks();
  std::list<std::string> get_list_of_chunks() { return m_list_chunks; }
  std::vector<std::string> get_vect_of_chunks();

private:
  const std::string m_path;

  std::list  <std::string> m_list_chunks;
  std::vector<std::string> m_vect_chunks;
 
  // Copy constructor and assignment are disabled by default
  FileFinder ( const FileFinder& ) ;
  FileFinder& operator = ( const FileFinder& ) ;

};

} // namespace PSXtcQC

#endif // PSXTCQC_FILEFINDER_H
