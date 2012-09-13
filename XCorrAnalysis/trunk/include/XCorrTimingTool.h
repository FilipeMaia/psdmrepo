#ifndef PSANA_SXR61612_XCORRTIMINGTOOL_H
#define PSANA_SXR61612_XCORRTIMINGTOOL_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XCorrTimingTool.
//
// Author: Sanne de Jong
//         adapted for psana by Ingrid Ofte
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
//#include "CCheader.h"
//#include "CCheader_array.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

#define MAXEVENTS 100000

namespace XCorrAnalysis {

/// @addtogroup XCorrAnalysis

/**
 *  @ingroup XCorrAnalysis
 *
 *  @brief Example module class for psana
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Sanne de Jong
 */

class XCorrTimingTool : public Module {
public:

  // Default constructor
  XCorrTimingTool (const std::string& name) ;

  // Destructor
  virtual ~XCorrTimingTool () ;

  /// Method which is called once at the beginning of the job
  virtual void beginJob(Event& evt, Env& env);
  
  /// Method which is called at the beginning of the run
  virtual void beginRun(Event& evt, Env& env);
  
  /// Method which is called at the beginning of the calibration cycle
  virtual void beginCalibCycle(Event& evt, Env& env);
  
  /// Method which is called with event data, this is the only required 
  /// method, all other methods are optional
  virtual void event(Event& evt, Env& env);
  
  /// Method which is called at the end of the calibration cycle
  virtual void endCalibCycle(Event& evt, Env& env);

  /// Method which is called at the end of the run
  virtual void endRun(Event& evt, Env& env);

  /// Method which is called once at the end of the job
  virtual void endJob(Event& evt, Env& env);

protected:

private:

  // Data members, this is for example purposes only
  long m_count;
  int m_runNr;

  Source m_opalSrc;    // Opal camera source
  
  std::string m_CCoutputfldr;       // file folder name for the background image
  std::string m_CCoutputnm;         // file name base for the background image
  std::string m_CCoutputSuffix;     // file extention for the background image
  
  int m_ccbackgroundrun;            // expects file "./CCoutputfldr/CCoutputnm%d.CCoutputSuffix" with %d being backgroundrun
  int m_singleshotout;              // writes the first # of single shots
  int m_initialguess;               // avaraged peak position (in pixels, estimated from the avaraged image)
  float m_ccmaxjump;                  // mmaximum allowed (+/-) deviation from initial guess (in ps)

  int m_signopal;
  
  //Regions of interest (in pixels ; 0 -> 1023)
  //SIGNAL:
  int m_sigROI_left;
  int m_sigROI_right;
  int m_sigROI_top;
  int m_sigROI_bottom;
  
  //BACKGROUND:
  int m_bgROI_left;
  int m_bgROI_right;
  int m_bgROI_top;
  int m_bgROI_bottom;
  

  float m_ccpixeltops;               // scaling factor ps per pixel
  double m_ccrotation;               // rotation of image in pixels per pixel-row; ie 0.1 = 5 deg. +ve = clockwise
  float m_ccpsoffset;                // offset value for the cross correlator time

  unsigned int m_ccsmoothbox;	     // number of pixels to smooth (sliding box avergage; choose odd number)
  unsigned int m_ccsmoothcycles;     // number of smoothing cycles
  float m_ccpeakthreshold;	     // only peaks in first derivative above this fraction of global max are taken into account 

  unsigned int m_ccwidth;            // size of image along horizontal
  unsigned int m_ccheight;           // size of image along vertical 
  unsigned int m_ccsize;

  //ndarray<uint16_t, 2> m_ccimage;  // copy of event image, apply normalizations & corrections to this one. 
  float* m_ccimage;                  // image array local copy
  float* m_ccbgimage;                // background image
  float* m_ccaverage;                // image array average

  float* m_ccint;                    // projection onto horizontal axis
  float* m_ccsmth;
  float* m_ccdiff;
  float CCeventarray[MAXEVENTS];
  
  
  void RotateCCImage();
  void NormalizeCCShot(); 

  float ReturnROI(int pix_x1, int pix_x2, int pix_y1, int pix_y2); // returns a number, the average value in ROI

/*   //Functions:----------------------------------------- */
/*   //--------------------------------------------------- */
/*   struct peaklist { */
/*     int ppos; */
/*     float pheight; */
/*   } ; */
  
/*   void InitCC(); */
/*   float getCCvalue(int eventcounter, int singleshotout, char filefolder[], int runnumber); */
/*   int LoadCCshot(int eventcounter); */
/*   float returnROI(float* input, int x1,  int x2, int y1, int y2, int width, int height); */
/*   void averagedimage(float input[], float output[], int ccsize); */
/*   int loadBGimage(float* output); */
/*   //  void cclineint(float input[], float output[], int width, int height, int sigROI_left, int sigROI_right, int sigROI_top, int sigROI_bottom); */
/*   void cclineint(float* input, float* output);//, int width, int height, int sigROI_left, int sigROI_right, int sigROI_top, int sigROI_bottom); */
/*   void killCC(); */
/*   void ccsmoothline(float input[], float output[], int size, int ccbox, int cccycles); */
/*   void ccdiffline(float input[], float output[], bool THflag); */
/*   float returnCCvalue(float input[], int size, int center, int center_offset,  int sbox, int scycles, int eventcounter); */
/*   int returnextreme(float input[], int start, int end, bool minmax); */
/*   float returnmax(float input[], int size, int start, int end, bool minmax); */
/*   int findpeaks(float input[] , peaklist output[], int size, int outputsize, int start, int end, float threshold, bool minmax); */
/*   void printArray(char name[], char comment[], int number, int input[]); */
/*   void printArray(char name[], char comment[], int number, float input[]); */
/*   void WriteImg( char filename[] , int version, int w, int h,  float* image); */
  
};

} // namespace XCorrAnalysis

#endif // PSANA_SXR61612_XCORRTIMINGTOOL_H
