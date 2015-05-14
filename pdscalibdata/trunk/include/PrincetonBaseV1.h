#ifndef PDSCALIBDATA_PRINCETONBASEV1_H
#define PDSCALIBDATA_PRINCETONBASEV1_H

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
//#include "pdsdata/psddl/princeton.ddl.h"

//-----------------------------

namespace pdscalibdata {

/**
 *  class PrincetonBaseV1 contains common parameters and methods for Princeton camera. 
 *
 *  This software was developed for the LCLS project.  
 *  If you use all or part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

class PrincetonBaseV1 {
public:

  typedef unsigned 	shape_t;
  typedef double  	cmod_t;

  const static size_t   Ndim = 2; 
  //const static size_t   Rows = 1300; 
  //const static size_t   Cols = 1340; 
  const static size_t   Rows = 0; // VARIABLE SHAPE DATA PARAMETERS WILL BE TAKEN FROM FILE METADATA
  const static size_t   Cols = 0; // VARIABLE SHAPE DATA PARAMETERS WILL BE TAKEN FROM FILE METADATA
  const static size_t   Size = Rows*Cols; 
  const static size_t   SizeCM = 7; 

  const shape_t* shape_base() { return &m_shape[0]; }
  const cmod_t*  cmod_base()  { return &m_cmod[0]; }
  const size_t   size_base()  { return Size; }

  ~PrincetonBaseV1 () {};

protected:

  PrincetonBaseV1 (){ 
    shape_t shape[Ndim]={Rows,Cols};            
    cmod_t cmod[SizeCM]={1,50,50,100,1,Size,1}; // use algorithm 1 to entire Princeton image
    std::memcpy(m_shape, &shape[0], sizeof(shape_t)*Ndim);
    std::memcpy(m_cmod,  &cmod[0],  sizeof(cmod_t)*SizeCM);
  };
  
private:
  shape_t m_shape[Ndim];
  cmod_t  m_cmod[SizeCM];
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_PRINCETONBASEV1_H
