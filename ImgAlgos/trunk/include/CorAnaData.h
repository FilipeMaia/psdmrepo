#ifndef IMGALGOS_CORANADATA_H
#define IMGALGOS_CORANADATA_H

//---------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CorAnaData.
//
//---------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <fstream>  // for ostream, ofstream
#include <iostream> // for cout, puts etc.

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

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *  @ingroup ImgAlgos
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

class CorAnaData  {
public:

  CorAnaData () ;
  virtual ~CorAnaData () ;

protected:
  void readMetadataFile();



private:

  std::ostream& m_olog;

  // Copy constructor and assignment are disabled by default
  CorAnaData ( const CorAnaData& ) ;
  CorAnaData& operator = ( const CorAnaData& ) ;
};

} // namespace ImgAlgos

#endif // IMGALGOS_CORANADATA_H
