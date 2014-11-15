#ifndef PDSCALIBDATA_CSPAD2X2BASEV2_H
#define PDSCALIBDATA_CSPAD2X2BASEV2_H

//------------------------------------------------------------------------
// File and Version Information:
//      $Revision$
//      $Id$
//      $HeadURL: https://pswww.slac.stanford.edu/svn/psdmrepo/pdscalibdata/trunk/include/CsPad2x2BaseV2.h $
//      $Date$
//
// Author: Mikhail Dubrovin
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
//#include <string>
#include <cstring>  // for memcpy

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
//#include "ndarray/ndarray.h"
//#include "pdsdata/psddl/andor.ddl.h"

//-----------------------------

namespace pdscalibdata {

/**
 *  class CsPad2x2BaseV2 contains common parameters and methods for CSPAD2x2 camera. 
 *
 *  This software was developed for the LCLS project.  
 *  If you use all or part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

class CsPad2x2BaseV2 {
public:

  typedef unsigned 	shape_t;
  typedef double  	cmod_t;

  const static size_t   Ndim = 3; 
  const static size_t   Segs = 2; 
  const static size_t   Rows = 185; 
  const static size_t   Cols = 388; 
  const static size_t   Size = Rows*Cols*Segs; 
  const static size_t   SizeCM = 4; 
  
 
  const shape_t* shape_base() { return &m_shape[0]; }
  const cmod_t*  cmod_base()  { return &m_cmod[0]; }
  const size_t   size_base()  { return Size; }

  ~CsPad2x2BaseV2 () {};

protected:

  CsPad2x2BaseV2 (){ 
    shape_t shape[Ndim]={Rows,Cols,Segs};            
    cmod_t cmod[SizeCM]={1, 25, 25, 100}; // use algorithm 1
    std::memcpy(m_shape, &shape[0], sizeof(shape_t)*Ndim);
    std::memcpy(m_cmod,  &cmod[0],  sizeof(cmod_t)*SizeCM);
  };
  
private:
  shape_t m_shape[Ndim];
  cmod_t  m_cmod[SizeCM];
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPAD2X2BASEV2_H
