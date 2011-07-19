#ifndef CSPADIMAGE_IMAGECSPAD_H
#define CSPADIMAGE_IMAGECSPAD_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImageCSPad.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "CSPadImage/CSPadCalibPars.h"
#include "CSPadImage/QuadParameters.h"
#include "CSPadImage/ImageCSPadQuad.h"
#include "CSPadImage/Image2D.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace CSPadImage {

/**
 *  @brief Example module class for psana
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class ImageCSPad: public Module {
public:
  enum { NRowsDet = 1765 };
  enum { NColsDet = 1765 };

  // Default constructor
  ImageCSPad(const std::string& name) ;

  // Destructor
  virtual ~ImageCSPad() ;

  /// Method which is called once at the beginning of the job
  //virtual void beginJob(Event& evt, Env& env);
  
  /// Method which is called at the beginning of the run
  virtual void beginRun(Event& evt, Env& env);
  
  /// Method which is called at the beginning of the calibration cycle
  //virtual void beginCalibCycle(Event& evt, Env& env);
  
  /// Method which is called with event data, this is the only required 
  /// method, all other methods are optional
  virtual void event(Event& evt, Env& env);
  
  /// Method which is called at the end of the calibration cycle
  //virtual void endCalibCycle(Event& evt, Env& env);

  /// Method which is called at the end of the run
  //virtual void endRun(Event& evt, Env& env);

  /// Method which is called once at the end of the job
  virtual void endJob(Event& evt, Env& env);

  void addQuadToCSPadImage(ImageCSPadQuad<uint16_t> *image_quad, QuadParameters *quadpars, CSPadCalibPars *cspad_calibpar);
  void fillQuadXYmin();
  Image2D<float>* getCSPadImage2D(){ return m_cspad_image_2d; } ;
  void saveCSPadImageInFile();
  void testOfImageClasses();

protected:

private:

  // Data members, this is for example purposes only
  
  Source m_src;         // Data source set from config file
  unsigned m_maxEvents;
  bool m_filter;
  long m_count;

  int m_Nquads;
  int m_Nrows;
  int m_Ncols;

  uint32_t m_roiMask        [4];
  uint32_t m_numAsicsStored [4];

  float    m_xmin_quad [4];
  float    m_ymin_quad [4];

  CSPadCalibPars *m_cspad_calibpar;

  float m_cspad_image[NRowsDet][NColsDet];

  Image2D<float> *m_cspad_image_2d;

};

} // namespace CSPadImage

#endif // CSPADIMAGE_IMAGECSPAD_H
