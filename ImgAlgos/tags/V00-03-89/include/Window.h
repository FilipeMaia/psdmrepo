#ifndef IMGALGOS_WINDOW_H
#define IMGALGOS_WINDOW_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Window.
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <stdint.h>    // uint8_t, uint32_t, etc.
#include <cstring>     // size_t
#include <sstream>     // for stringstream
#include <algorithm>   // std::max
#include <iostream>    // std::ostream
#include <iomanip>     // for setw, setfill

//#include <string>
//#include <fstream>   // ofstream
//#include <sstream>   // for stringstream
//#include <iostream>
//#include <math.h>
//#include <stdio.h>

#include "MsgLogger/MsgLogger.h"

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *  @ingroup ImgAlgos
 *
 *  @brief Holds 2-d window limits and pointer to the window shape.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

using namespace std;

//--------------------

class Window {

 public: 

  // Size limits [MIN_SIZE_LIMIT, MAX_SIZE_LIMIT) 
  static const size_t MIN_SIZE_LIMIT = 0;  
  static const size_t MAX_SIZE_LIMIT = 100000;  

  typedef unsigned shape_t;

  size_t segind;
  size_t rowmin;
  size_t rowmax;
  size_t colmin;
  size_t colmax;

//--------------------
  /// Constructor (including default) sets window parameters from shape or from default limits
  Window(const shape_t* shape=0) 
  { 
    set(shape);
  }

//--------------------
  /// Constructor sets specified window parameters and checks their validity if shape is is already available
  Window( const size_t& segi
        , const size_t& rmin
        , const size_t& rmax
        , const size_t& cmin
        , const size_t& cmax )
  : m_shape(0)
  {
    set(segi, rmin, rmax, cmin, cmax);
  }

//--------------------
  /// Set window parameters
  void set( const size_t& segi
          , const size_t& rmin
          , const size_t& rmax
          , const size_t& cmin
          , const size_t& cmax )
  {
    segind = segi;
    rowmin = rmin;
    rowmax = rmax;
    colmin = cmin;
    colmax = cmax;
    if(m_shape) validate(m_shape);
  }

//--------------------
  /// Set window parameters from shape
  void set(const shape_t* shape=0)
  {
    m_shape = shape; 
    rowmin = MIN_SIZE_LIMIT;
    rowmax = (shape) ? m_shape[0] : MAX_SIZE_LIMIT;
    colmin = MIN_SIZE_LIMIT;
    colmax = (shape) ? m_shape[1] : MAX_SIZE_LIMIT;
  }

//--------------------
  /// Validate window parameters
  void validate(const shape_t* shape) 
  {
    m_shape = shape;
    rowmin = std::max((int)MIN_SIZE_LIMIT, int(rowmin));
    rowmax = std::min((size_t) m_shape[0], rowmax);
    colmin = std::max((int)MIN_SIZE_LIMIT, int(colmin));
    colmax = std::min((size_t) m_shape[1], colmax);
  }

//--------------------
  /// Returns pointer to the shape
  const shape_t* shape() { return m_shape; }

//--------------------
  /// Prints memeber data
  void print()
  {
    std::stringstream ss; 
    ss << "\nsegind  : " << segind
       << "\nrowmin  : " << rowmin
       << "\nrowmax  : " << rowmax
       << "\ncolmin  : " << colmin
       << "\ncolmax  : " << colmax;  
    if(m_shape) ss << "\nshape[0]  : " << m_shape[0]
                   << "\nshape[1]  : " << m_shape[1];
    ss << '\n';

    MsgLog(_name(), info, ss.str()); 
  }

//--------------------
  /// Make window with specified parameters
  Window& make( const size_t& segi
              , const size_t& rmin
              , const size_t& rmax
              , const size_t& cmin
              , const size_t& cmax )
  {
    set(segi, rmin, rmax, cmin, cmax);
    m_shape = 0;
    return *this;
  }


//--------------------
  Window(const Window& rhs)
  {
    segind = rhs.segind;
    rowmin = rhs.rowmin;
    rowmax = rhs.rowmax;
    colmin = rhs.colmin;
    colmax = rhs.colmax;
  }

//--------------------
  /// copy object operator
  Window& operator=(const Window& rhs)
  {
    segind = rhs.segind;
    rowmin = rhs.rowmin;
    rowmax = rhs.rowmax;
    colmin = rhs.colmin;
    colmax = rhs.colmax;
    //m_shape= rhs.shape(); 
    return *this;
  }

//--------------------

 private:

  const shape_t* m_shape;  

  /// Returns string name of the class for messanger
  inline const char* _name() {return "ImgAlgos::Window";}
};

//--------------------

  /// operator <<
  inline std::ostream& operator<<(std::ostream& out, const Window& w) {
    return out << "  segind:" << std::setw(2) << w.segind 
               << "  rowmin:" << std::setw(4) << w.rowmin 
               << "  rowmax:" << std::setw(4) << w.rowmax                       
               << "  colmin:" << std::setw(4) << w.colmin 
               << "  colmax:" << std::setw(4) << w.colmax;
  }

//--------------------
//--------------------
//--------------------
//--------------------
//--------------------

} // namespace ImgAlgos

#endif // IMGALGOS_WINDOW_H
