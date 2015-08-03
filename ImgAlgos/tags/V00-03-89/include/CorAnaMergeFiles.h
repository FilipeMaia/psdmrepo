#ifndef IMGALGOS_CORANAMERGEFILES_H
#define IMGALGOS_CORANAMERGEFILES_H

//---------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CorAnaMergeFiles.
//
//---------------------------------

//-----------------
// C/C++ Headers --
//-----------------
//#include <string>
//#include <vector>
#include <stdint.h>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ImgAlgos/CorAna.h"
#include "ImgAlgos/TimeInterval.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *  @ingroup ImgAlgos
 *
 *  @brief Merge the input (block vs tau-index) to output (image vs tau-index)
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class CorAnaMergeFiles : public CorAna {
public:

  typedef double hist_t;

  CorAnaMergeFiles () ;
  virtual ~CorAnaMergeFiles () ;

protected:
  void openFiles();
  void closeFiles();
  void mergeFiles();

private:

  TimeInterval*  m_timer1;

  std::ifstream* p_inp;    // pointer to input array of input files 
  std::ofstream  p_out;    // pointer to output file 

  // Copy constructor and assignment are disabled by default
  CorAnaMergeFiles ( const CorAnaMergeFiles& ) ;
  CorAnaMergeFiles& operator = ( const CorAnaMergeFiles& ) ;
};

} // namespace ImgAlgos

#endif // IMGALGOS_CORANAMERGEFILES_H
