//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Utility functions for calculating the X-ray timing jitter correction 
//      from the cross-correlation image. 
//
// Author: Sanne de Jong
//         adapted for psana by Ingrid Ofte
//------------------------------------------------------------------------

namespace CCTime 
{
  void normalizeccshot(float* image, int ccwidth, int ccheight, int signopal, 
		       int bgROI_left, int bgROI_right, int bgROI_top,  int bgROI_bottom, 
		       int sigROI_left, int sigROI_right, int sigROI_top, int sigROI_bottom); 

  float returnROI(float* input, int x1,  int x2, int y1, int y2, int width, int height); 
}

// Leave it at this (working) test for now. 
// I think it makes more sense to integrate 
// all of this into the psana module that own
// the parameters. This lessens the need for 
// argument passing. 




/* static char CCoutputfldr[256] = "./CCfiles/";                   //file folder name for the background image */
/* static char CCoutputnm[256] = "Background";                     //file name base for the background image */
/* static char CCoutputSuffix[25] = ".img";                        //file extention for the background image */

/* int ccbackgroundrun = 127;                                      //expects file "./CCoutputfldr/CCoutputnm%d.CCoutputSuffix" with %d being backgroundrun */

/* int singleshotout = 3;                                          //writes the first # of single shots */

/* int initialguess = 440;                                         //avaraged peak position (in pixels, estimated from the avaraged image) */
/* const float ccmaxjump = 0.4;                            //maximum allowed (+/-) deviation from initial guess (in ps) */

/* //Regions of interest (in pixels ; 0 -> 1023):-------- */
/* //SIGNAL: */
/* int sigROI_left = 270; */
/* int sigROI_right = 540; */
/* int sigROI_top = 600; */
/* int sigROI_bottom = 940; */

/* //BACKGROUND: */
/* int bgROI_left = 0; */
/* int bgROI_right = 200; */
/* int bgROI_top = 0; */
/* int bgROI_bottom = 200; */

/* const FrameDetector cccamera = SxrBeamlineOpal1;                //Camera name for the Cross Correlator */
/* int signopal = 1; */

/* float ccpixeltops = .01075;                                     //scaling factor ps per pixel */
/* float ccrotation = 0.04;                                                //rotation of image in pixels per pixel-row; ie 0.1 = 5 deg. +ve = clockwise */
/* float ccpsoffset = 0;                                           //offset value for the cross correlator time */

/* bool cccrosscheckflag = 1;                                      //Set to 1 to perform edge detection on 2 halfs of the ROI and compare outcome. */
/*                                                                                         //NOTE: camera rotation angle correction has to be applied correctly! */

/* const int ccsmoothbox = 5;                                      //number of pixels to smooth (sliding box avergage; choose odd number) */
/* const int ccsmoothcycles = 5;                                   //number of smoothing cycles */
/* float ccpeakthreshold = 0.5;                                    //only peaks in first derivative above this fraction of global max are taken into account */



/* #ifndef u_short */
/* #define u_short unsigned short */
/* #endif */

/* //ARRAYS and Variables:------------------------------ */
/* //--------------------------------------------------- */
/* float* ccaverage = (float *)0; */
/* u_short* ccrawimage = (u_short*)0; */
/* float* ccimage = (float *)0; */
/* float* ccbgimage = (float *)0; */
/* float* ccint = (float *)0; */
/* float* ccsmth = (float *)0; */
/* float* ccdiff = (float *)0; */
/* #define maxevents 100000 */
/* float CCeventarray[maxevents] = {0}; */

/* int ccsize = 0; */
/* int cccounter = {0}; */
/* bool ccbgavailable = false; */
/* bool ccshotfail = false; */

/* //Functions:----------------------------------------- */
/* //--------------------------------------------------- */
/* struct peaklist { */
/*  	int ppos; */
/*  	float pheight; */
/* } ; */

/* void InitCC(); */
/* float getCCvalue(int eventcounter, int singleshotout, char filefolder[], int runnumber); */
/* int LoadCCshot(int eventcounter); */
/* void normalizeccshot(float* image, int imagesize, int bgROI_left, int bgROI_right, int bgROI_top,  int bgROI_bottom, */
/* 				int sigROI_left, int sigROI_right, int sigROI_top, int sigROI_bottom); */
/* void averagedimage(float input[], float output[], int ccsize); */
/* int loadBGimage(float* output); */
/* void cclineint(float input[], float output[], int width, int height, int sigROI_left, int sigROI_right, int sigROI_top, int sigROI_bottom); */
/* void killCC(); */
/* void ccsmoothline(float input[], float output[], int size, int ccbox, int cccycles); */
/* void ccdiffline(float input[], float output[], bool THflag); */
/* float returnCCvalue(float input[], int size, int center, int center_offset,  int sbox, int scycles, int eventcounter); */
/* int returnextreme(float input[], int start, int end, bool minmax); */
/* float returnmax(float input[], int size, int start, int end, bool minmax); */
/* int findpeaks(float input[] , peaklist output[], int size, int outputsize, int start, int end, float threshold, bool minmax); */
/* void printArray(char name[], char comment[], int number, int input[]); */
/* void printArray(char name[], char comment[], int number, float input[]); */
/* void WriteImg( char filename[] , int version, int w, int h,  float* image); */
/* void rotateccimage(float* image, int width, int height, float rotationfactor); */
/* float cccoreprocs(int val_top, int val_bottom, int eventcounter); */



/* #ifndef u_short */
/* #define u_short unsigned short */
/* #endif */

/* //ARRAYS and Variables:------------------------------ */
/* //--------------------------------------------------- */
/* float* ccaverage = (float *)0; */
/* u_short* ccrawimage = (u_short*)0; */
/* float* ccimage = (float *)0; */
/* float* ccbgimage = (float *)0; */
/* float* ccint = (float *)0; */
/* float* ccsmth = (float *)0; */
/* float* ccdiff = (float *)0; */
/* #define maxevents 100000 */
/* float CCeventarray[maxevents] = {0}; */

/* int ccsize = 0; */
/* int cccounter = {0}; */
/* bool ccbgavailable = false; */
/* bool ccshotfail = false; */

/* //Functions:----------------------------------------- */
/* //--------------------------------------------------- */
/* struct peaklist { */
/*  	int ppos; */
/*  	float pheight; */
/* } ; */

/* void InitCC(); */
/* float getCCvalue(int eventcounter, int singleshotout, char filefolder[], int runnumber); */
/* int LoadCCshot(int eventcounter); */
/* float returnROI(float* input, int x1,  int x2, int y1, int y2, int width, int height); */
/* void averagedimage(float input[], float output[], int ccsize); */
/* int loadBGimage(float* output); */
/* void cclineint(float input[], float output[], int width, int height, int sigROI_left, int sigROI_right, int sigROI_top, int sigROI_bottom); */
/* void killCC(); */
/* void ccsmoothline(float input[], float output[], int size, int ccbox, int cccycles); */
/* void ccdiffline(float input[], float output[], bool THflag); */
/* float returnCCvalue(float input[], int size, int center, int center_offset,  int sbox, int scycles, int eventcounter); */
/* int returnextreme(float input[], int start, int end, bool minmax); */
/* float returnmax(float input[], int size, int start, int end, bool minmax); */
/* int findpeaks(float input[] , peaklist output[], int size, int outputsize, int start, int end, float threshold, bool minmax); */
/* void printArray(char name[], char comment[], int number, int input[]); */
/* void printArray(char name[], char comment[], int number, float input[]); */
/* void WriteImg( char filename[] , int version, int w, int h,  float* image); */
/* void rotateccimage(float* image, int width, int height, float rotationfactor); */
/* float cccoreprocs(int val_top, int val_bottom, int eventcounter); */


