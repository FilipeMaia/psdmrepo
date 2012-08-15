#ifndef CSPADPIXCOORDS_QUADPARAMETERS_H
#define CSPADPIXCOORDS_QUADPARAMETERS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class QuadParameters.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <vector>
#include <stdint.h>

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

namespace CSPadPixCoords {

/// @addtogroup CSPadPixCoords

/**
 *  @ingroup CSPadPixCoords
 *
 *  @brief QuadParameters class holds current parameters for the CSPad quads.
 *  
 *  Object of this class is used for passing of the list of current quad parameters. 
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see PixCoords2x1, PixCoordsQuad, PixCoordsCSPad
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class QuadParameters  {
public:

  // Default constructor
  QuadParameters () ;

  // Regular constructor
  /**
   *  @brief Creates an object which holds the current quad parameters.
   *  
   *  @param[in] quadNumber      Current quad number.
   *  @param[in] nrows           Number of rows reserved for the quad image.
   *  @param[in] ncols           Number of columns reserved for the quad image.
   *  @param[in] numAsicsStored  Number of ASICs stored for this quad in the event. 
   *  @param[in] roiMask         8-bit mask showing the sections presented in data.
   */ 
  QuadParameters (uint32_t         quadNumber, 
                  size_t           nrows=850, 
                  size_t           ncols=850, 
                  uint32_t         numAsicsStored=16, 
                  uint32_t         roiMask=255) ;

  // Destructor
  virtual ~QuadParameters () ;

  size_t           getNRows             (){ return m_nrows;   };
  size_t           getNCols             (){ return m_ncols;   };
  uint32_t         getRoiMask           (){ return m_roiMask; };
  uint32_t         getQuadNumber        (){ return m_quadNumber; };
  uint32_t         getNumberAsicsStroed (){ return m_numAsicsStored; };

  void print ();

private:

  // Copy constructor and assignment are disabled by default
  QuadParameters ( const QuadParameters& ) ;
  QuadParameters operator = ( const QuadParameters& ) ;

//------------------
// Static Members --
//------------------

  // Data members
  uint32_t         m_quadNumber;
  size_t           m_nrows;
  size_t           m_ncols;
  uint32_t         m_numAsicsStored;
  uint32_t         m_roiMask;

};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_QUADPARAMETERS_H
