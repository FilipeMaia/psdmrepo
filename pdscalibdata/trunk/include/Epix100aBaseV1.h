#ifndef PDSCALIBDATA_EPIX100ABASEV1_H
#define PDSCALIBDATA_EPIX100ABASEV1_H

//------------------------------------------------------------------------
// File and Version Information:
//      $Revision$
//      $Id$
//      $HeadURL$
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
 *  class Epix100aBaseV1 contains common parameters and methods for Andor camera. 
 *
 *  This software was developed for the LCLS project.  
 *  If you use all or part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

class Epix100aBaseV1 {
public:

  typedef unsigned 	shape_t;
  typedef double  	cmod_t;

  const static size_t   Ndim = 2; 
  const static size_t   Rows = 704; 
  const static size_t   Cols = 768; 
  const static size_t   Size = Rows*Cols; 
  const static size_t   SizeCM = 7; 

  const shape_t* shape_base() { return &m_shape[0]; }
  const cmod_t*  cmod_base()  { return &m_cmod[0]; }
  const size_t   size_base()  { return Size; }

  ~Epix100aBaseV1 () {};

protected:

  Epix100aBaseV1 (){ 
    shape_t shape[Ndim]={Rows,Cols};            
    cmod_t cmod[SizeCM]={1,50,50,100,1,Size,1}; // use algorithm 1 to entire image
    std::memcpy(m_shape, &shape[0], sizeof(shape_t)*Ndim);
    std::memcpy(m_cmod,  &cmod[0],  sizeof(cmod_t)*SizeCM);
  };
  
private:
  shape_t m_shape[Ndim];
  cmod_t  m_cmod[SizeCM];
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_EPIX100ABASEV1_H
