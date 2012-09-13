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

#include "XCorrAnalysis/CCTime.h"

namespace CCTime 
{
  
  //This function makes the normalized CC image:
  void normalizeccshot(float* image, int ccwidth, int ccheight, int signopal, 
		       int bgROI_left, int bgROI_right, int bgROI_top,  int bgROI_bottom,
		       int sigROI_left, int sigROI_right, int sigROI_top, int sigROI_bottom)
  {
    float bgvalue = 0;
    bgvalue = returnROI(image, bgROI_left, bgROI_right, bgROI_top, bgROI_bottom, ccwidth, ccheight);
    float sigvalue = 0;
    sigvalue = returnROI(image, sigROI_left, sigROI_right, sigROI_top, sigROI_bottom, ccwidth, ccheight);
    
    for(int i = 0 ; i < (ccwidth*ccheight) ; i++){
      image[i] -= bgvalue;
      image[i] /= signopal*(sigvalue-bgvalue);
    }
    
  }
  
  
  //Returns the averaged value for a set region of interest:
  float returnROI(float* input, int x1,  int x2, int y1, int y2, int width, int height)
  {
    //	float* testimage = (float*)0;
    //	testimage = new float[width*height];
    
    int numpoints = (x2 - x1 + 1) * (y2 - y1 +1);
    float roival = 0;
    int size = width*height;
    for (int i = 0 ; i < (x2 - x1 + 1); i++){
      for (int k = 0 ; k < (y2 - y1 + 1); k++){
	if( (y1 + i + (y1 + k) * width) < size)
	  roival += input[y1 + i + (y1 + k) * width];
	//			testimage[y1 + i + (y1 + k) * width] = 1;
      }
    }
    roival /= numpoints;
    
    //	static char FileName[256];
    //	sprintf(FileName,"%sROItest%s",CCoutputfldr, CCoutputSuffix);
    //	WriteImg( FileName, 1, width, height, testimage);
    //	delete testimage;
    
    return roival;
  }
  
  // Leave it at this (working) test for now. 
  // I think it makes more sense to integrate 
  // all of this into the psana module that own
  // the parameters. This lessens the need for 
  // argument passing. 
  
}









// //initialize some required arrays:
// void InitCC()
// {
// 	int mkfolder = mkdir( CCoutputfldr, 0777);
// 	if (mkfolder == 0){
// 		printf ( "Created folder %s\n", CCoutputfldr);
// 	}
// 	ccsize = ccwidth * ccheight;

// 	ccrawimage = new u_short[ccsize];
// 	ccaverage = new float [ccsize];
// 		for(int i = 0; i < ccsize ; i++) ccaverage[i] = 0;
// 	ccimage = new float [ccsize];
// 	ccbgimage = new float [ccsize];

// 	ccint = new float [(sigROI_right - sigROI_left)+1];
// 	ccsmth = new float [(sigROI_right - sigROI_left)+1];
//  	ccdiff = new float [(sigROI_right - sigROI_left)+1];
// }

// //Procedures that form the heart of the X-correlator:
// float cccoreprocs(int val_top, int val_bottom, int eventcounter)
// {
// 	float returntime;
// 	//Do the integration of the ROI:
// 	cclineint(ccimage , ccint, ccwidth, ccheight, sigROI_left, sigROI_right, val_top, val_bottom);
// 	//Smooth the integral:
// 	ccsmoothline(ccint, ccsmth, (sigROI_right - sigROI_left +1), ccsmoothbox, ccsmoothcycles);
// 	//Diffrentiate:
// 	ccdiffline(ccsmth, ccdiff, 1);
// 	//return the time value from a peak evaluation:
// 	returntime = returnCCvalue(ccdiff, (sigROI_right - sigROI_left +1), initialguess, sigROI_left, ccsmoothbox, ccsmoothcycles, eventcounter);
// 	return returntime;
// }


// //call for myana code:
// float getCCvalue(int eventcounter, int singleshotout, char filefolder[], int runnumber)
// {
// 	float time = 1000000;
// 	int failccload = 1;
// 	failccload = LoadCCshot(eventcounter);
// 	if (failccload == 0){

// 		rotateccimage(ccimage, ccwidth, ccheight, ccrotation);

// 		normalizeccshot(ccimage, ccsize, bgROI_left,  bgROI_right, bgROI_top, bgROI_bottom,
// 					sigROI_left,  sigROI_right, sigROI_top, sigROI_bottom);
// 		for(int i = 0 ; i < ccsize ; i++) ccimage[i] -= ccbgimage[i];

// 		//build avarage CC image:
// 		averagedimage(ccimage, ccaverage, ccsize);

// 		//return corrected time:
// 		switch (cccrosscheckflag){
// 		case 0:
// 			time = cccoreprocs(sigROI_top, sigROI_bottom, eventcounter);
// 			break;
// 		case 1:
// 			int yvalmiddle = sigROI_top + (sigROI_bottom - sigROI_top)/2;
// 			float timetry1 = cccoreprocs(sigROI_top, yvalmiddle, eventcounter);
// 			float timetry2 = cccoreprocs(yvalmiddle, sigROI_bottom, eventcounter);
// 			if(fabs(timetry1 - timetry2) < (ccmaxjump/2) && (timetry1 <100000) &&  (timetry2 <100000)){
// 				time = (timetry1 + timetry2)/2;
// 			}
// 			else{
// 				printf("Event#: %d; CC fail; cross-check t1 = %03f, t2 = %03f ps.\n", eventcounter, timetry1, timetry2);
// 			}
// 			break;
// 		}

// 		//Write a few single shots, if desired:
// 		if (eventcounter < singleshotout){
// 			char FileName[256];
// 			sprintf(FileName,"%sCC_run%d_sht%03d.img",filefolder, runnumber, eventcounter);
// 			WriteImg( FileName, 1, ccwidth, ccheight, ccimage );

// 			char comment[256] = "";
// //			sprintf(FileName,"%sCC_run%d_sht%03d_lineInt.txt",filefolder, runnumber, eventcounter);
// //			printArray(FileName, comment, ((sigROI_right - sigROI_left)+1), ccint);

// 			sprintf(FileName,"%sCC_run%d_sht%03d_lineIntSmth.txt",filefolder, runnumber, eventcounter);
// 			printArray(FileName, comment, ((sigROI_right - sigROI_left)+1), ccsmth);

// //			sprintf(FileName,"%sCC_run%d_sht%03d_diff.txt",filefolder, runnumber, eventindex);
// //			printArray(FileName, comment, ((sigROI_right - sigROI_left)+1), ccdiff);
// 		}
// 	}
// 	CCeventarray[eventcounter] = time;
// 	return time;
// }

// //load an opal shot:
// int LoadCCshot(int eventcounter)
// {

// 	if(ccbgavailable == 0){
// 		//Get image
// 		int opalfail = 0;
// 		opalfail = getOpal1kValue(cccamera, ccwidth, ccheight, ccrawimage);
// 		if (opalfail) {
// 			printf("Event# %d : cannot read Opal image.\n", eventcounter);
// 			return 1;
// 		}
// 		else{
// 			for (int i = 0 ; i< ccsize ; i++) ccimage[i] = static_cast<float>( ccrawimage[i] );
// 			return 0;
// 		}
// 	}
// 	else{
// 		printf("Event# %d : CC fail: background image not loaded.\n", eventcounter);
// 		return 1;
// 	}
// }


// //Performes a rotation;
// void rotateccimage(float* image, int width, int height, float rotationfactor)
// {
// 	if(rotationfactor != 0){
// 		int tempsize = width*height;
// 		float* tempimagebuffer  = (float *)0;
// 		tempimagebuffer = new float [tempsize];

// //		printf("rotating image\n");

// 		for(int i = 0 ; i <tempsize ; i++){
// 			tempimagebuffer[i] = image[i];
// 		}

// 		for(int i = 0 ; i <tempsize ; i++){
// 			float tempcounter = static_cast<float>(i);
// 			int curpix = static_cast <int> (tempcounter + (tempcounter * rotationfactor / width));
// 			if (curpix < tempsize && curpix >= 0){
// //			printf("i = %d curpix = %d\t", i, curpix);
// 				image[i] = tempimagebuffer[curpix];
// 			}
// 		}
// //		printf("\n");
// 	delete [] tempimagebuffer;
// 	}
// }



// //Makes the sum for an averaged image:
// void averagedimage(float input[], float output[], int size)
// {
// 	for(int i =0 ; i < size ; i++){
// 		output[i] += input[i];
// 	}
// }


// //This function loads the BG image for the cross correlator:
// int loadBGimage(float* output)
// {
// 	static char bgImageName[256];
// 	sprintf(bgImageName, "%s%s%d%s", CCoutputfldr, CCoutputnm, ccbackgroundrun, CCoutputSuffix);

// 	unsigned int lengthbuf;
// 	FILE * filebuffer = NULL;
// 	filebuffer = fopen(bgImageName, "r");


// 	printf("--------------\n");
// 	//Error message if BG run file does not excist:
// 	if (filebuffer == NULL)
//     {
// 		printf("Cannot open background image file %s.\n", bgImageName);
// 		printf("--------------\n");
// 		return -1;
//     }
// 	//Error message if BG run file has wrong format:
// 	fseek(filebuffer, 0, SEEK_END);
// 	lengthbuf = ftell(filebuffer);
// 	printf("Filesize = %d.\n", lengthbuf);
// 	rewind(filebuffer);
// 	if (lengthbuf != 3* sizeof(short) + sizeof(float)*ccwidth*ccheight)
//     {
// 		printf("Wrong background images size for file %s.\n", bgImageName);
// 		printf("--------------\n");
// 		return 1;
//     }

// 	printf("Succesfully loaded background image file %s.\n", bgImageName);
// 	printf("--------------\n");

// 	fseek(filebuffer, 6, SEEK_SET);
// 	fread(output, sizeof(float), ccsize, filebuffer);
// 	//rotate the BG image:
// 	rotateccimage(output, ccwidth, ccheight, ccrotation);
// 	return 0;
// }


// //Makes a 1D line integral over a specified 2D ROI:
// void cclineint(float input[], float output[], int width, int height, int sigROI_left, int sigROI_right, int sigROI_top, int sigROI_bottom)
// {
// 	int currentpoint = 0;
// 	int numpnts_x = (sigROI_right - sigROI_left) +1;
// 	int numpnts_y = (sigROI_bottom - sigROI_top) +1;

// 	for(int i = 0 ; i <  numpnts_x ; i++){
// 		output[i] = 0;
// 	}

// 	for(int i = 0 ; i <  numpnts_y ; i++){
// 		for(int k = 0 ; k < numpnts_x ; k++){
// 			currentpoint = (sigROI_left + k) + (sigROI_top + i)*width;
// 			if( currentpoint < ccsize) output[k] += input[currentpoint];
// 		}
// 	}
// }


// //Does a smooting with a set boxsize and number of cycles over a 1D array:
// void ccsmoothline(float input[], float output[], int size, int ccbox, int cccycles)
// {
// 	if (ccbox % 2 == 0){
// 		ccbox +=1;
// //		printf ("smoothing box size even; increased by 1.\n");
// 	}


// 	int startval = (ccbox-1)/2;
// 	int endval = size - (ccbox-1)/2;

// 	float ccboxval = 0;

// 	float* inputbuffer = (float*)0;
// 	inputbuffer = new float[size];
// 	float* outputbuffer = (float*)0;
// 	outputbuffer = new float[size];

// 	for(int i = 0; i< size ; i++) inputbuffer[i] = input[i];
// 	for(int i = 0; i< size ; i++) outputbuffer[i] = 0;


// 	//loops over the number of smooth cycles:
// 	for(int i = 0; i< cccycles ; i++){

// 		for(int k = startval; k < endval ; k++){

// 			int bstart = k-(ccbox-1)/2;
// 			int bend = k+1+(ccbox-1)/2;
// 			//printf ("point %d: %d -> %d.\n", k, bstart, bend);

// 			ccboxval = 0;

// 			for(int l = bstart; l< bend ; l++){
// 				ccboxval += inputbuffer[l];
// 			//	printf ("%f +..", inputbuffer[l]);
// 			}
// 			//printf (" = %f. \n", ccboxval);
// 			ccboxval /= ccbox;
// 			outputbuffer[k] = ccboxval;
// 		}
// 		for(int k = 0 ; k< startval ; k++) outputbuffer[k] = outputbuffer[startval];
// 		for(int k = endval ; k< size ; k++) outputbuffer[k] = outputbuffer[endval-1];
// 		if (cccycles > 1){
// 			for(int k = 0; k< size ; k++) inputbuffer[k] = outputbuffer[k];
// 		}

// 	}
// 	for(int k = 0; k< size ; k++) output[k] = outputbuffer[k];
// 	delete [] inputbuffer;
// 	delete [] outputbuffer;
// }

// //Makes a simple (forward) differential of a 1D array:
// void ccdiffline(float input[], float output[], bool THFlag)
// {
// 	float diffvalue;
// 	int numpnts_x = (sigROI_right - sigROI_left) +1;
// 	for(int i = 0; i< numpnts_x-1 ; i++){
// 		diffvalue = input[i+1] - input[i];

// 		if (THFlag ==1 ){
// 			//threshold the differential; negative peaks are not the edge.
// 			//Values below 0 are set to 0.
// 			if (diffvalue < 0) output[i] = 0;
// 			else output[i] = diffvalue;
// 		}
// 		else output[i] = diffvalue;

// 	}
// 	ccdiff[numpnts_x-1] = ccdiff[numpnts_x-2];
// }

// //Does the actual edge determination:
// float returnCCvalue(float input[], int size, int center, int center_offset, int sbox, int scycles, int eventcounter)
// {
// 	//Allowed number of peaks:
// 	int peaksallowed = 4;
// //	float minpeakwidth = 0.070;	//allowed edge width in fs
// //	float maxpeakwidth = 0.35;

// 	float return_time = 1000000;
// 	float estimatedtime = ccpixeltops * initialguess;

// 	peaklist* allpeaks = (peaklist*)0;
// 	allpeaks = new peaklist[peaksallowed];
// 	for (int i = 0 ; i< peaksallowed ; i++){
// 		 allpeaks[i].ppos = 0;
// 		 allpeaks[i].pheight = 0;
// 	}

// 	//find the global maximum in the first derivative:
// 	float globalmax = returnmax(input, size , 0, size-1, 0);

// 	//only peaks above threshold are taken into account:
// 	int numpeaks = findpeaks(input, allpeaks, size, peaksallowed, 0, size-1, ccpeakthreshold*globalmax, 0);

// 	//too many peaks edges above threshold detected:
// 	if (numpeaks > peaksallowed) {
// 		printf("Event#: %d; CC fail: more then %d edges above threshold detected.\n", eventcounter, peaksallowed);
// 		delete [] allpeaks;
// 		return return_time;
// 	}
// 	//no edges found in CC shot:
// 	if (numpeaks == 0){
// 		printf("Event#: %d; CC fail: no edges detected.\n", eventcounter);
// 		delete [] allpeaks;
// 		return return_time;
// 	}
// 	if (numpeaks > 0 && numpeaks < peaksallowed){
// //		printf("Event#: %d; CC detected %d edge(s).\n", eventcounter, numpeaks);
// 		//Check whether there is an edge within the set time window:
// 		int goodpeak = 0;
// 		float edgetime = 0;
// 		for (int i = 0 ; i< numpeaks ; i++){
// 			edgetime = (static_cast<float>(allpeaks[i].ppos) + static_cast<float>(center_offset)) * ccpixeltops + ccpsoffset;
// 			//check whether the peak is within the set limits
// 			if (edgetime > estimatedtime-ccmaxjump && edgetime < estimatedtime+ccmaxjump){
// 				return_time = edgetime;
// 				goodpeak += 1;
// 			}
// 		}
// 		if (goodpeak == 0){
// 			printf("Event#: %d; CC fail: no edges within time limits detected.\n", eventcounter);
// 			delete [] allpeaks;
// 			return return_time;
// 		}
// 		if (goodpeak >1) {
// 			printf("Event#: %d; CC fail: %d high edges within time limits detected.\n", eventcounter, goodpeak);
// 			return_time = 1000000;
// 			delete [] allpeaks;
// 			return return_time;
// 		}
// 		//Only one peak good peak within the set time window:
// 		if (goodpeak ==1){
// //			printf("Event#: %d; edge found at pixel: %d.\n", eventcounter, static_cast<int>(return_time / ccpixeltops - center_offset));
// 			//additional constraint could be the peak width... but not iplemented...
// 			delete [] allpeaks;
// 			return return_time;
// 		}
// 	}
// 	return return_time;
// }


// //Finds all peaks above threshold in range between start and end:
// int findpeaks(float input[] , peaklist output[] , int size, int outputsize, int start, int end, float threshold, bool minmax)
// {
// 	float lastval = 0;
// 	float currentvalue = 0;
// 	int peakhistory[6] = {0,0,0,0,0,0};
// 	int peakcount = 0;

// 	//only implement max finder:
// 	for (int i = start; i < end ; i++ ){
// 		for (int k = 0; k < 5 ; k++ ) peakhistory[k] = peakhistory [k+1];
// 		lastval = currentvalue;
// 		currentvalue = input[i];

// 		if (currentvalue < threshold){
// 			lastval = threshold;
// 			peakhistory[5] = 2;
// 		}
// 		else{
// 			if (currentvalue > lastval) peakhistory[5] = 1;
// 			if (currentvalue < lastval) peakhistory[5] = -1;

// 			//peak detected:
// 			if (peakhistory[0] == 1 && peakhistory[1] == 1 && peakhistory[2] == 1 && peakhistory[3] == -1 && peakhistory[4] == -1 && peakhistory[5] == -1 ){
// 				if (peakcount < outputsize){
// 					output[peakcount].ppos =  i-3;
// 					output[peakcount].pheight = input[i-3];
// 					peakcount +=1;
// //					printf("Peak %d detected: position = %d, height = %f\n", peakcount, (i-3), (input[i-3]));
// 				}
// 				else{
// 					peakcount +=1;
// 					return peakcount;
// 				}
// 			}
// 		}
// 	}
// 	return peakcount;
// }


// //returns the overall position of the max or min value:
// //bool minmax ; (0) = max ; (1) = min.
// float returnmax(float input[], int size, int start, int end, bool minmax){

// 	float peakval = 0;
// 	if (end > size) end = size-1;

// 	switch (minmax){
// 		case 0:
// 			peakval = 0;
// 			for( int i = start ; i < end+1 ; i++){
// 				if(input[i]> peakval){
// 					peakval = input[i];
// 				}
// 			}
// 			break;
// 		case 1:
// 			peakval = 100000000;
// 			for( int i = start ; i < end+1 ; i++){
// 				if(input[i]< peakval){
// 					peakval = input[i];
// 				}
// 			}
// 			break;
// 	}
// 	return peakval;
// }


// void wrap_up_cc(int eventindex, int runnumber, char outputfolder[])
// {
// 	//Make and write average image:
// 	for(int i = 0 ; i < ccsize ; i++) ccaverage[i] /= eventindex;
// 	char FileName[256];
// 	sprintf(FileName,"%sCC_run%d_AVG%dshts.img",outputfolder, runnumber, eventindex);
// 	WriteImg( FileName, 1, ccwidth, ccheight, ccaverage );

// 	printf("--------------\n");
// 	printf("Averaged CC image for run %d over %d shots.\n", runnumber, eventindex);
// 	printf("Written CC image %s to file.\n", FileName);

// 	sprintf(FileName,"%sCC_run%d_events.txt",outputfolder, runnumber);
// 	char comment[256] = "time(ps)\n";
// 	printArray(FileName, comment, eventindex, CCeventarray);


// 	//get the average edge position from the average:
// 	cclineint(ccaverage , ccint, ccwidth, ccheight, sigROI_left, sigROI_right, sigROI_top, sigROI_bottom);
// 	ccsmoothline(ccint, ccsmth, (sigROI_right - sigROI_left +1), ccsmoothbox, ccsmoothcycles);
// 	ccdiffline(ccsmth, ccdiff, 1);
// 	float time;
// 	time = returnCCvalue(ccdiff, (sigROI_right - sigROI_left +1), initialguess, sigROI_left, ccsmoothbox, ccsmoothcycles, eventindex);


// 	//get the average edge position from single shots:
// 	int badshots = 0;
// 	int goodshots = eventindex;
// 	float avgtime_ss = 0;
// 	for(int i = 0 ; i < eventindex ; i++){
// 		if (CCeventarray[i] >= 100000){
// 			badshots += 1;
// 			goodshots -= 1;
// //			printf("event %d ; bad shot...\n", i);
// 		}
// 		if (CCeventarray[i] != 1000000) avgtime_ss += CCeventarray[i];

// 	}
// 	avgtime_ss /= goodshots;
// 	int avg_pix_ss = static_cast<int> ((avgtime_ss - ccpsoffset) / ccpixeltops);

// 	float perc_bad_shots = 100 * (static_cast<float>(badshots) / static_cast<float>(eventindex));

// 	printf("Number of bad CC events = %d out of %d, (%.1f %%) \n", badshots, eventindex, perc_bad_shots);
// 	printf("Average peak position from individual single shots = %.3f (pixel-#: %d).\n", avgtime_ss, avg_pix_ss);

// 	avg_pix_ss = static_cast<int> ((time - ccpsoffset) / ccpixeltops);
// 	printf("Average peak position from total average = %.3f (pixel-#: %d).\n", time, avg_pix_ss);

// 	printf("--------------\n");
// }

// void killCC()
// {
// 	delete [] ccaverage;
// 	delete [] ccimage;
// 	delete [] ccbgimage;

// 	delete [] ccint;
// 	delete [] ccsmth;
// 	delete [] ccdiff;

// 	printf("Deleted CC dynamic memory.\n");
// //	delete [] ccrawimage;
// }

// /*Writes an image in the designated folder:
// - first 6 bytes are shorts: version, width and height
// - next array is 1024*1024 floats containing the image*/
// void WriteImg( char filename[] , int version, int w, int h,  float* image)
// {
// 	FILE* imgFile = fopen(filename, "wb");
// 	unsigned short size[3];
// 	size[0] = (unsigned short) version;
// 	size[1] = (unsigned short) w;
// 	size[2] = (unsigned short) h;

// 	fwrite(size, 2, 3 , imgFile);
// 	fwrite(image, 4, (w*h) , imgFile);
// }

// //Prints all the diagnostics arrays:
// void printArray(char name[], char comment[], int number, float input[])
// {
// 	FILE* FileWrite = fopen(name, "w" );
// 	fprintf (FileWrite, "%s\n", comment);

// 	for ( int l = 0; l < number; l++ ){
// 		fprintf (FileWrite, "%f\n", input[l]);
// 	}

// 	fclose(FileWrite);
// 	//printf("Printed file: %s.\n", name);
// }
// void printArray(char name[], char comment[], int number, int input[])
// {
// 	FILE* FileWrite = fopen(name, "w" );
// 	fprintf (FileWrite, "%s\n", comment);

// 	for ( int l = 0; l < number; l++ ){
// 		fprintf (FileWrite, "%d\n", input[l]);
// 	}

// 	fclose(FileWrite);
// 	//printf("Printed file: %s.\n", name);
// }
