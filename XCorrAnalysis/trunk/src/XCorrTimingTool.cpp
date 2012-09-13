//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XCorrTimingTool...
//
// Author List:
//      Sanne de Jong  (Original)
//      Ingrid Ofte    (Adaption to the psana framework)
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XCorrAnalysis/XCorrTimingTool.h"

//-----------------
// C/C++ Headers --
//-----------------
//#include <sys/dir.h>
//#include <sys/types.h>
#include <sys/stat.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

// Include detector data headers from psddl_psana package:
#include "psddl_psana/opal1k.ddl.h"
#include "psddl_psana/camera.ddl.h"

#include "PSEvt/EventId.h"

#include "XCorrAnalysis/CCTime.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace XCorrAnalysis;
PSANA_MODULE_FACTORY(XCorrTimingTool)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XCorrAnalysis {
//----------------
// Constructors --
//----------------
XCorrTimingTool::XCorrTimingTool (const std::string& name)
  : Module(name)
  , m_opalSrc()
{
  // get the values from configuration or use defaults
  m_opalSrc = configStr("opalSrc", "DetInfo(:Opal1000)");  
  m_CCoutputfldr = configStr("CCoutputfldr","./CCfiles/");
  m_CCoutputnm = configStr("CCoutputnm","Background");
  m_CCoutputSuffix = configStr("CCoutputSuffix",".img");

  m_ccbackgroundrun = config("ccbackgroundrun",7);
  m_singleshotout = config("singleshotout",3);
  m_initialguess = config("initialguess",500);
  m_ccmaxjump = config("ccmaxjump",0.4);
  m_signopal = config("signopal",-1);

  m_sigROI_left = config("sigROI_left",320);
  m_sigROI_right = config("sigROI_right",650);
  m_sigROI_top = config("sigROI_top",430);
  m_sigROI_bottom = config("sigROI_bottom",700);
  m_bgROI_left = config("bgROI_left",0);
  m_bgROI_right = config("bgROI_right",200);
  m_bgROI_top = config("bgROI_top",0);
  m_bgROI_bottom = config("bgROI_bottom",200);

  m_ccpixeltops = config("ccpixeltops",0.01075);
  m_ccrotation = config("ccrotation", 0.04);
  m_ccpsoffset = config("ccpoffset",0);
  m_ccsmoothbox = config("ccsmoothbox",5);
  m_ccsmoothcycles = config("ccsmoothcycles",5);
  m_ccpeakthreshold = config("ccpeakthreshold", 0.5);
}

//--------------
// Destructor --
//--------------
XCorrTimingTool::~XCorrTimingTool ()
{
}

/// Method which is called once at the beginning of the job
void 
XCorrTimingTool::beginJob(Event& evt, Env& env)
{
  m_count = 0;

  int mkfolder = mkdir( m_CCoutputfldr.data() , 0777);
  if (mkfolder == 0){
    MsgLog(name(),info, "Created folder " << m_CCoutputfldr );
  }

  // Get the configuration for the camera
  shared_ptr<Psana::Opal1k::ConfigV1> config = env.configStore().get(m_opalSrc);
  if (!config.get()) {
    MsgLog(name(), info, "No configuration for Opal found ... ");
  }
    
  WithMsgLog(name(), debug, str) {
    str << "Psana::Opal1k::ConfigV1:";
    str << "\n  black_level = " << config->black_level();
    str << "\n  gain_percent = " << config->gain_percent();
    str << "\n  output_resolution = " << config->output_resolution();
    str << "\n  vertical_binning = " << config->vertical_binning();
    str << "\n  output_mirroring = " << config->output_mirroring();
    str << "\n  vertical_remapping = " << int(config->vertical_remapping());
    str << "\n  output_offset = " << config->output_offset();
    str << "\n  output_resolution_bits = " << config->output_resolution_bits();
    str << "\n  defect_pixel_correction_enabled = " << int(config->defect_pixel_correction_enabled());
    str << "\n  output_lookup_table_enabled = " << int(config->output_lookup_table_enabled());
    
    if (config->output_lookup_table_enabled()) {
      const ndarray<uint16_t, 1>& output_lookup_table = config->output_lookup_table();
      str << "\n  output_lookup_table =";
      for (unsigned i = 0; i < output_lookup_table.size(); ++ i) {
	str << ' ' << output_lookup_table[i];
      }
    }
    if (config->number_of_defect_pixels()) {
      str << "\n  defect_pixel_coordinates =";
      const ndarray<Psana::Camera::FrameCoord, 1>& coord = config->defect_pixel_coordinates();
      for (unsigned i = 0; i < coord.size(); ++ i) {
	str << "(" << coord[i].column() << ", " << coord[i].row() << ")";
      }
    }
  }
  
  m_ccwidth = config->Row_Pixels;
  m_ccheight = config->Column_Pixels;
  m_ccsize =  m_ccwidth * m_ccheight;

  // Allocate memory for a new image array, floats:
  m_ccimage = new float[m_ccsize];
  m_ccaverage = new float [m_ccsize];
  m_ccbgimage = new float [m_ccsize];
  for(unsigned i = 0; i < m_ccsize ; i++) {
    m_ccimage[i] = 0.;
    m_ccaverage[i] = 0.;
    m_ccbgimage[i] = 0.;
  }
  
  unsigned width = (m_sigROI_right - m_sigROI_left)+1;
  m_ccint = new float[width];
  m_ccsmth = new float[width];
  m_ccdiff = new float[width];
  

}

/// Method which is called at the beginning of the run
void 
XCorrTimingTool::beginRun(Event& evt, Env& env)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    m_runNr = eventId->run();
  }

  //Initialize x-correlator:
  //InitCC();
  
  //load the cc background image:
  //bool ccbgavailable = loadBGimage(m_ccbgimage);
  //if(ccbgavailable == 0){
  //  normalizeccshot(m_ccbgimage);
  // }
}

/// Method which is called at the beginning of the calibration cycle
void 
XCorrTimingTool::beginCalibCycle(Event& evt, Env& env)
{ 
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
XCorrTimingTool::event(Event& evt, Env& env)
{
//  Process Opal Frame
  shared_ptr<Psana::Camera::FrameV1> opal = evt.get(m_opalSrc);

  float cctime = 1000000;  
  if (!opal.get()){
    std::cout << "No opal " << std::endl;
  } else {
    
    const unsigned ndim = 2;
    const ndarray<uint16_t, ndim>& evtimage = opal->data16();

    // Copy over all the values
    for (unsigned int i(0); i<m_ccsize;i++){
      m_ccimage[i] = float(evtimage.data()[i]);
    }

    // Rotate image
    RotateCCImage();
    
    // Normalize
    NormalizeCCShot();

    //ccint = CCLineInt(m_ccimage)

    
//     // Subtract the background image
//     for(unsigned i = 0 ; i < m_ccsize ; i++) m_ccimage[i] -= m_ccbgimage[i];
    
//     // Build avarage CC image:
//     averagedimage(m_ccimage, m_ccaverage, m_ccsize);
    
//     //Do the integration of the ROI (fills the 1D array m_ccint) :
//     cclineint(m_ccimage, m_ccint);// ccwidth, ccheight, sigROI_left, sigROI_right, sigROI_top, sigROI_bottom);

//     //Smooth the integral:
//     ccsmoothline(ccint, ccsmth, (sigROI_right - sigROI_left +1), ccsmoothbox, ccsmoothcycles);

//     //Diffrentiate:
//     ccdiffline(ccsmth, ccdiff, 1);

//     // return the time value from a peak evaluation:
//     cctime = returnCCvalue(ccdiff, (sigROI_right - sigROI_left +1), initialguess, sigROI_left, ccsmoothbox, ccsmoothcycles, m_count);
    
//     //Write a few single shots, if desired:
//     if (m_count < singleshotout){
//       char FileName[256];
//       sprintf(FileName,"%sCC_run%d_sht%03ld.img",m_CCoutputfldr, m_runNr, m_count);
//       WriteImg( FileName, 1, ccwidth, ccheight, ccimage );
      
//       char comment[256] = "";
//       //                      sprintf(FileName,"%sCC_run%d_sht%03d_lineInt.txt",m_CCoutputfldr, m_runNr, m_count);
//       //                      printArray(FileName, comment, ((sigROI_right - sigROI_left)+1), ccint);
      
//       sprintf(FileName,"%sCC_run%d_sht%03ld_lineIntSmth.txt",m_CCoutputfldr, m_runNr, m_count);
//       printArray(FileName, comment, ((sigROI_right - sigROI_left)+1), ccsmth);
      
//       //                      sprintf(FileName,"%sCC_run%d_sht%03d_diff.txt",m_CCoutputfldr, m_runNr, eventindex);
//       //                      printArray(FileName, comment, ((sigROI_right - sigROI_left)+1), ccdiff);
//     }
  }
  //CCeventarray[m_count] = cctime;
  
  //printf("XCorrTime = %.4f \n",cctime );
  
  
  // Give time to the event as a shared pointer
  //shared_ptr<float> cctime_ptr(new float(cctime));
  //evt.put(cctime_ptr,"XCorrTimingTool:time");
  
  
  // increment event counter
  ++ m_count;
}

/// Method which is called at the end of the calibration cycle
void 
XCorrTimingTool::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
XCorrTimingTool::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
XCorrTimingTool::endJob(Event& evt, Env& env)
{
}


// ///
// /// Utility functions: 
// ///


//Performs a rotation 
// ==> ASK! What is rotatingfactor?? I've changed the name to angle... 
void XCorrTimingTool::RotateCCImage()
{
  if(m_ccrotation == 0) return;
  
  float* tempimagebuffer  = (float*)0;
  tempimagebuffer = new float [m_ccsize];
  
  for(unsigned i = 0 ; i <m_ccsize ; i++){
    tempimagebuffer[i] = m_ccimage[i];
  }
  
  for(unsigned i = 0 ; i <m_ccsize ; i++){
    float tempcounter = static_cast<float>(i);
    int curpix = static_cast <int> (tempcounter + (tempcounter * m_ccrotation / m_ccwidth));
    if (curpix < (int)m_ccsize && curpix >= 0){
      //                      printf("i = %d curpix = %d\t", i, curpix);
      m_ccimage[i] = tempimagebuffer[curpix];
    }
  }
  //              printf("\n");
  delete [] tempimagebuffer;
}



void XCorrTimingTool::NormalizeCCShot()
{
  // This function makes the normalized CC image:
  float bgvalue = 0;
  bgvalue = ReturnROI(m_bgROI_left, m_bgROI_right, m_bgROI_top, m_bgROI_bottom);
  float sigvalue = 0;
  sigvalue = ReturnROI( m_sigROI_left, m_sigROI_right, m_sigROI_top, m_sigROI_bottom);
  
  for(unsigned i = 0 ; i < m_ccsize ; i++){
    m_ccimage[i] -= bgvalue;
    m_ccimage[i] /= m_signopal*(sigvalue-bgvalue);
  }
  
}


float XCorrTimingTool::ReturnROI(int x1, int x2, int y1, int y2)
{
  // Returns the averaged value for a set region of interest:
  int nx = 1 + (x2 - x1); // x == first index == row
  int ny = 1 + (y2 - y1); // y = second index == column

  float roival = 0;
  for (int ix = 0 ; ix <nx; ix++){ // loop over rows
    for (int iy = 0 ; iy < ny; iy++){ // loop over columns
      if( (y1 + ix + (y1 + iy) * m_ccwidth) < m_ccsize)
 	roival += m_ccimage[y1 + ix + (y1 + iy) * m_ccwidth];
      //       testimage[y1 + ix + (y1 + iy) * m_ccwidth] = 1;
    }
  }
  roival /= (nx*ny);
  
//   //      static char FileName[256];
//   //      sprintf(FileName,"%sROItest%s",m_CCoutputfldr, m_CCoutputSuffix);
//   //      WriteImg( FileName, 1, width, height, testimage);
//   //      delete testimage;
  
  return roival;
}



// //Makes the sum for an averaged image:
// void averagedimage(float input[], float output[], int size)
// {
//   for(int i =0 ; i < size ; i++){
//     output[i] += input[i];
//   }
// }



// //This function loads the BG image for the cross correlator:
// int loadBGimage(float* output)
// {
//   static char bgImageName[256];
//   sprintf(bgImageName, "%s%s%d%s", m_CCoutputfldr, m_CCoutputnm, ccbackgroundrun, m_CCoutputSuffix);
  
//   unsigned int lengthbuf;
//   FILE * filebuffer = NULL;
//   filebuffer = fopen(bgImageName, "r");
  
  
//   printf("--------------\n");
//   //Error message if BG run file does not excist:
//   if (filebuffer == NULL)
//     {
//       printf("Cannot open background image file %s.\n", bgImageName);
//       printf("--------------\n");
//       return -1;
//     }
//   //Error message if BG run file has wrong format:
//   fseek(filebuffer, 0, SEEK_END);
//   lengthbuf = ftell(filebuffer);
//   printf("Filesize = %d.\n", lengthbuf);
//   rewind(filebuffer);
//   if (lengthbuf != 3* sizeof(short) + sizeof(float)*ccwidth*ccheight)
//     {
//       printf("Wrong background images size for file %s.\n", bgImageName);
//       printf("--------------\n");
//       return 1;
//     }
  
//   printf("Succesfully loaded background image file %s.\n", bgImageName);
//   printf("--------------\n");
  
//   fseek(filebuffer, 6, SEEK_SET);
//   fread(output, sizeof(float), ccsize, filebuffer);
//   return 0;
// }
  
  
// void cclineint(const float* input, float* output)
// {
//   int currentpoint = 0;
//   int numpnts_x = (m_sigROI_right - m_sigROI_left) +1;
//   int numpnts_y = (m_sigROI_bottom - m_sigROI_top) +1;
  
//   for(int i = 0 ; i <  numpnts_x ; i++){
//     output[i] = 0.;
//   }
  
//   for(int i = 0 ; i <  numpnts_y ; i++){
//     for(int k = 0 ; k < numpnts_x ; k++){
//       currentpoint = (m_sigROI_left + k) + (m_sigROI_top + i)*n_ccwidth;
//       if( currentpoint < m_ccsize) output[k] += input[currentpoint];
//     }
//   }
// }
  

// //Does a smooting with a set boxsize and number of cycles over a 1D array:
// void ccsmoothline(float input[], float output[], int size, int ccbox, int cccycles)
// {
//   if (ccbox % 2 == 0){
//     ccbox +=1;
//     //              printf ("smoothing box size even; increased by 1.\n");
//   }
  
  
//   int startval = (ccbox-1)/2;
//   int endval = size - (ccbox-1)/2;
  
//   float ccboxval = 0;
  
//   float* inputbuffer = (float*)0;
//   inputbuffer = new float[size];
//   float* outputbuffer = (float*)0;
//   outputbuffer = new float[size];
  
//   for(int i = 0; i< size ; i++) inputbuffer[i] = input[i];
//   for(int i = 0; i< size ; i++) outputbuffer[i] = 0;
  
  
//   //loops over the number of smooth cycles:
//   for(int i = 0; i< cccycles ; i++){
    
//     for(int k = startval; k < endval ; k++){
      
//       int bstart = k-(ccbox-1)/2;
//       int bend = k+1+(ccbox-1)/2;
//       //printf ("point %d: %d -> %d.\n", k, bstart, bend);
      
//       ccboxval = 0;
      
//       for(int l = bstart; l< bend ; l++){
// 	ccboxval += inputbuffer[l];
// 	//      printf ("%f +..", inputbuffer[l]);
//       }
//       //printf (" = %f. \n", ccboxval);
//       ccboxval /= ccbox;
//       outputbuffer[k] = ccboxval;
//     }
//     for(int k = 0 ; k< startval ; k++) outputbuffer[k] = outputbuffer[startval];
//     for(int k = endval ; k< size ; k++) outputbuffer[k] = outputbuffer[endval-1];
//     if (cccycles > 1){
//       for(int k = 0; k< size ; k++) inputbuffer[k] = outputbuffer[k];
//     }
    
//   }
//   for(int k = 0; k< size ; k++) output[k] = outputbuffer[k];
//   delete [] inputbuffer;
//   delete [] outputbuffer;
// }



// //Makes a simple (forward) differential of a 1D array:
// void ccdiffline(float input[], float output[], bool THFlag)
// {
//   float diffvalue;
//   int numpnts_x = (sigROI_right - sigROI_left) +1;
//   for(int i = 0; i< numpnts_x-1 ; i++){
//     diffvalue = input[i+1] - input[i];
    
//     if (THFlag ==1 ){
//       //threshold the differential; negative peaks are not the edge.
//       //Values below 0 are set to 0.
//       if (diffvalue < 0) output[i] = 0;
//       else output[i] = diffvalue;
//     }
//     else output[i] = diffvalue;
    
//   }
//   ccdiff[numpnts_x-1] = ccdiff[numpnts_x-2];
// }



// //Does the actual edge determination:
// float returnCCvalue(float input[], int size, int center, int center_offset, int sbox, int scycles, int eventcounter){

//   //Allowed number of peaks:
//   int peaksallowed = 4;
//   //      float minpeakwidth = 0.070;     //allowed edge width in fs
//   //      float maxpeakwidth = 0.35;
  
//   float return_time = 1000000;
//   float estimatedtime = ccpixeltops * initialguess;
  
//   peaklist* allpeaks = (peaklist*)0;
//   allpeaks = new peaklist[peaksallowed];
//   for (int i = 0 ; i< peaksallowed ; i++){
//     allpeaks[i].ppos = 0;
//     allpeaks[i].pheight = 0;
//   }
  
//   //find the global maximum in the first derivative:
//   float globalmax = returnmax(input, size , 0, size-1, 0);
  
//   //only peaks above threshold are taken into account:
//   int numpeaks = findpeaks(input, allpeaks, size, peaksallowed, 0, size-1, ccpeakthreshold*globalmax, 0);
  
//   //too many peaks edges above threshold detected:
//   if (numpeaks > peaksallowed) {
//     printf("Event#: %d; CC fail: more then %d edges above threshold detected.\n", eventcounter, peaksallowed);
//     delete [] allpeaks;
//     return return_time;
//   }
  
//   //no edges found in CC shot:
//   if (numpeaks == 0){
//     printf("Event#: %d; CC fail: no edges detected.\n", eventcounter);
//     delete [] allpeaks;
//     return return_time;
//   }
//   if (numpeaks > 0 && numpeaks < peaksallowed){
//     //              printf("Event#: %d; CC detected %d edge(s).\n", eventcounter, numpeaks);
//     //Check whether there is an edge within the set time window:
//     int goodpeak = 0;
//     float edgetime = 0;
//     for (int i = 0 ; i< numpeaks ; i++){
//       edgetime = (static_cast<float>(allpeaks[i].ppos) + static_cast<float>(center_offset)) * ccpixeltops + ccpsoffset;
//       //check whether the peak is within the set limits
//       if (edgetime > estimatedtime-ccmaxjump && edgetime < estimatedtime+ccmaxjump){
// 	return_time = edgetime;
// 	goodpeak += 1;
//       }
//     }
//     if (goodpeak == 0){
//       printf("Event#: %d; CC fail: no edges within time limits detected.\n", eventcounter);
//       delete [] allpeaks;
//       return return_time;
//     }
//     if (goodpeak >1) {
//       printf("Event#: %d; CC fail: %d high edges within time limits detected.\n", eventcounter, goodpeak);
//       return_time = 1000000;
//       delete [] allpeaks;
//       return return_time;
//     }
//     //Only one peak good peak within the set time window:
//     if (goodpeak ==1){
//       //printf("Event#: %d; edge found at pixel: %d.\n", eventcounter, static_cast<int>(return_time / ccpixeltops - center_offset));
//       //additional constraint could be the peak width... but not iplemented...
//       delete [] allpeaks;
//       return return_time;
//     }
//   }
//   return return_time;     
// }


// //Finds all peaks above threshold in range between start and end:
// int findpeaks(float input[] , peaklist output[] , int size, int outputsize, int start, int end, float threshold, bool minmax)
// {
//   float lastval = 0;
//   float currentvalue = 0;
//   int peakhistory[6] = {0,0,0,0,0,0};
//   int peakcount = 0;
  
//   //only implement max finder:
//   for (int i = start; i < end ; i++ ){
//     for (int k = 0; k < 5 ; k++ ) peakhistory[k] = peakhistory [k+1];
//     lastval = currentvalue;
//     currentvalue = input[i];
    
//     if (currentvalue < threshold){
//       lastval = threshold;
//       peakhistory[5] = 2;
//     }
//     else{
//       if (currentvalue > lastval) peakhistory[5] = 1;
//       if (currentvalue < lastval) peakhistory[5] = -1;
      
//       //peak detected:
//       if (peakhistory[0] == 1 && peakhistory[1] == 1 && peakhistory[2] == 1 && peakhistory[3] ==
// 	  -1 && peakhistory[4] == -1 && peakhistory[5] == -1 ){
// 	if (peakcount < outputsize){
// 	  output[peakcount].ppos =  i-3;
// 	  output[peakcount].pheight = input[i-3];
// 	  peakcount +=1;
// 	  //                                      printf("Peak %d detected: position = %d, height = %f\n", peakcount, (i-3), (input[i-3]));
// 	}
// 	else{
// 	  peakcount +=1;
// 	  return peakcount;
// 	}
//       }
//     }
//   }
//   return peakcount;
// }

// //returns the overall position of the max or min value:
// //bool minmax ; (0) = max ; (1) = min.
// float returnmax(float input[], int size, int start, int end, bool minmax){

//   float peakval = 0;
//   if (end > size) end = size-1;
  
//   switch (minmax){
//   case 0:
//     peakval = 0;
//     for( int i = start ; i < end+1 ; i++){
//       if(input[i]> peakval){
// 	peakval = input[i];
//       }
//     }
//     break;
//   case 1:
//     peakval = 100000000;
//     for( int i = start ; i < end+1 ; i++){
//       if(input[i]< peakval){
// 	peakval = input[i];
//       }
//     }
//     break;
//   }
//   return peakval;
// }



// void wrap_up_cc(int eventindex, int runnumber, char outputfolder[])
// {
//   //Make and write average image:
//   for(int i = 0 ; i < ccsize ; i++) ccaverage[i] /= eventindex;
//   char FileName[256];
//   sprintf(FileName,"%sCC_run%d_AVG%dshts.img",outputfolder, runnumber, eventindex);
//   WriteImg( FileName, 1, ccwidth, ccheight, ccaverage );
  
//   printf("--------------\n");
//   printf("Averaged CC image for run %d over %d shots.\n", runnumber, eventindex);
//   printf("Written CC image %s to file.\n", FileName);
  
//   sprintf(FileName,"%sCC_run%d_events.txt",outputfolder, runnumber);
//   char comment[256] = "time(ps)\n";
//   printArray(FileName, comment, eventindex, CCeventarray);
  
  
//   //get the average edge position from the average:
//   cclineint(ccaverage , ccint, ccwidth, ccheight, sigROI_left, sigROI_right, sigROI_top, sigROI_bottom);
//   ccsmoothline(ccint, ccsmth, (sigROI_right - sigROI_left +1), ccsmoothbox, ccsmoothcycles);
//   ccdiffline(ccsmth, ccdiff, 1);
//   float time = 
//     returnCCvalue(ccdiff, (sigROI_right - sigROI_left +1), initialguess, sigROI_left, ccsmoothbox, ccsmoothcycles, eventindex);
  
  
//   //get the average edge position from single shots:
//   int badshots = 0;
//   int goodshots = eventindex;
//   float avgtime_ss = 0;
//   for(int i = 0 ; i < eventindex ; i++){
//     if (CCeventarray[i] == 1000000){
//       badshots += 1;
//       goodshots -= 1;
//       //                      printf("event %d ; bad shot...\n", i);
//     }
//     if (CCeventarray[i] != 1000000) avgtime_ss += CCeventarray[i];
    
//   }
//   avgtime_ss /= goodshots;
//   int avg_pix_ss = static_cast<int> ((avgtime_ss - ccpsoffset) / ccpixeltops);
  
//   float perc_bad_shots = 100 * (static_cast<float>(badshots) / static_cast<float>(eventindex));
  
//   printf("Number of bad CC events = %d out of %d, (%.1f %%) \n", badshots, eventindex, perc_bad_shots);
//   printf("Average peak position from individual single shots = %.3f (pixel-#: %d).\n", avgtime_ss, avg_pix_ss);
  
//   avg_pix_ss = static_cast<int> ((time - ccpsoffset) / ccpixeltops);
//   printf("Average peak position from total average = %.3f (pixel-#: %d).\n", time, avg_pix_ss);
  
//   printf("--------------\n");
// }




// void killCC()
// {
//   delete [] ccaverage;
//   delete [] ccimage;
//   delete [] ccbgimage;
  
//   delete [] ccint;
//   delete [] ccsmth;
//   delete [] ccdiff;
  
//   printf("Deleted CC dynamic memory.\n");
//   //      delete [] ccrawimage;
// }


// /*Writes an image in the designated folder:
// - first 6 bytes are shorts: version, width and height
// - next array is 1024*1024 floats containing the image*/
// void WriteImg( char filename[] , int version, int w, int h,  float* image)
// {
//   FILE* imgFile = fopen(filename, "wb");
//   unsigned short size[3];
//   size[0] = (unsigned short) version;
//   size[1] = (unsigned short) w;
//   size[2] = (unsigned short) h;
  
//   fwrite(size, 2, 3 , imgFile);
//   fwrite(image, 4, (w*h) , imgFile);
// }


// //Prints all the diagnostics arrays:
// void printArray(char name[], char comment[], int number, float input[])
// {
//   FILE* FileWrite = fopen(name, "w" );
//   fprintf (FileWrite, "%s\n", comment);
  
//   for ( int l = 0; l < number; l++ ){
//     fprintf (FileWrite, "%f\n", input[l]);
//   }
  
//   fclose(FileWrite);
//   //printf("Printed file: %s.\n", name);
// }
// void printArray(char name[], char comment[], int number, int input[])
// {
//   FILE* FileWrite = fopen(name, "w" );
//   fprintf (FileWrite, "%s\n", comment);
  
//   for ( int l = 0; l < number; l++ ){
//     fprintf (FileWrite, "%d\n", input[l]);
//   }
  
//   fclose(FileWrite);
//   //printf("Printed file: %s.\n", name);
// }

} // namespace XCorrAnalysis


