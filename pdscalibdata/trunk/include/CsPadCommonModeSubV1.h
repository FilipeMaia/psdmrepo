#ifndef PDSCALIBDATA_CSPADCOMMONMODESUBV1_H
#define PDSCALIBDATA_CSPADCOMMONMODESUBV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadCommonModeSubV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
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

namespace pdscalibdata {

/**
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class CsPadCommonModeSubV1  {
public:

  enum CommonMode { 
    None = 0,
    Default = 1
  };
  
  enum { DataSize = 16 };

  // constant for unknown common mode
  enum { UnknownCM = -10000 };
  
  /// Default constructor, all pixel codes set to 0
  CsPadCommonModeSubV1 () ;
  
  /**
   *  Read constant from file.
   *  
   *  @throw std::exception
   */
  CsPadCommonModeSubV1 (const std::string& fname) ;

  /**
   *  Initialize constants from parameters.
   */
  CsPadCommonModeSubV1 (CommonMode mode, const double data[DataSize]) ;

  // Destructor
  ~CsPadCommonModeSubV1 () ;

  // access common mode
  CommonMode mode() const { return CommonMode(m_mode); }

  // access common mode data
  const double* data() const { return m_data; }

  /**
   *  Find common mode for an CsPad  section.
   *  
   *  Function will return UnknownCM value if the calculation 
   *  cannot be performed (or need not be performed).
   *  
   *  @param sdata  pixel data
   *  @param peddata  pedestal data, can be zero pointer
   *  @param pixStatus  pixel status data, can be zero pointer
   *  @param ssize  size of all above arrays
   *  @param stride increment for pixel indices
   */ 
  float findCommonMode(const int16_t* sdata,
                       const float* peddata, 
                       const  uint16_t *pixStatus, 
                       unsigned ssize,
                       int stride = 1) const;
  
protected:

private:

  // Data members  
  uint32_t m_mode;
  double m_data[DataSize];

  // Copy constructor and assignment are disabled by default
  CsPadCommonModeSubV1 ( const CsPadCommonModeSubV1& ) ;
  CsPadCommonModeSubV1& operator = ( const CsPadCommonModeSubV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPADCOMMONMODESUBV1_H
