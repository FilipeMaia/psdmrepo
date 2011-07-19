//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImageCSPadQuad...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadImage/ImageCSPadQuad.h"
#include "CSPadImage/ImageCSPad2x1.h"
#include "CSPadImage/Image2D.h"

//-----------------
// C/C++ Headers --
//-----------------


#include <iostream> // for cout
#include <fstream>
#include <boost/lexical_cast.hpp>
#include <time.h>

//#include <string>
using namespace std;

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace CSPadImage {

//----------------
// Constructors --
//----------------

template <typename T>
ImageCSPadQuad<T>::ImageCSPadQuad (const T* data, QuadParameters* quadpars, CSPadCalibPars *cspad_calibpar) :
  m_data(data),
  m_quadpars(quadpars),
  m_cspad_calibpar(cspad_calibpar)
{
  cout << "Here in ImageCSPadQuad<T>::ImageCSPadQuad" << endl;

      cout << "Start timer" << endl;
      struct timespec start, stop;
      int status = clock_gettime( CLOCK_REALTIME, &start ); // Get LOCAL time

  fillQuadImage();

      status = clock_gettime( CLOCK_REALTIME, &stop ); // Get LOCAL time
      cout << "Time to fill quad is " 
           << stop.tv_sec - start.tv_sec + 1e-9*(stop.tv_nsec - start.tv_nsec) 
           << " sec" << endl;

  saveQuadImageInFile();
}

//----------------

template <typename T>
void ImageCSPadQuad<T>::fillQuadImage ()
{
  //v_image_shape = m_quadpars -> getImageShapeVector();
  uint32_t quad    = m_quadpars -> getQuadNumber();
  uint32_t roiMask = m_quadpars -> getRoiMask();
                   //m_quadpars -> print();

        v_image_shape  = m_quadpars -> getImageShapeVector();
        //m_cspad_calibpar -> printCalibPars();

        //cout << "roiMask = " << roiMask << endl;
        m_n2x1         = v_image_shape[0];        // 8
        m_nrows2x1     = v_image_shape[1];        // 185
        m_ncols2x1     = v_image_shape[2];        // 388
	m_sizeOf2x1Img = m_nrows2x1 * m_ncols2x1; // 185*388;

	for (uint32_t sect=0; sect<m_n2x1; ++sect)
	  {
            bool bitIsOn = roiMask & (1<<sect);
	    if( !bitIsOn ) continue;

	    float xcenter  = m_cspad_calibpar -> getQuadMargX  ()
                           + m_cspad_calibpar -> getCenterX    (quad,sect)
	                   + m_cspad_calibpar -> getCenterCorrX(quad,sect);

	    float ycenter  = m_cspad_calibpar -> getQuadMargY  ()
                           + m_cspad_calibpar -> getCenterY    (quad,sect)
	                   + m_cspad_calibpar -> getCenterCorrY(quad,sect);

	    float zcenter  = m_cspad_calibpar -> getQuadMargZ  ()
                           + m_cspad_calibpar -> getCenterZ    (quad,sect)
	                   + m_cspad_calibpar -> getCenterCorrZ(quad,sect);

	    float rotation = m_cspad_calibpar -> getRotation   (quad,sect);

	    addSectionToQuadImage(sect, xcenter, ycenter, zcenter, rotation);
	  }


    m_quad_image_2d = new Image2D<T>(&m_quad_image[0][0],NRows,NCols);  

}

//----------------

template <typename T>
void ImageCSPadQuad<T>::addSectionToQuadImage(uint32_t sect, float xcenter, float ycenter, float zcenter, float rotation)
{
  // cout << "ImageCSPadQuad<T>::addSectionToQuadImage():  sect=" << sect << endl;

            ImageCSPad2x1<uint16_t>* image_2x1 = new ImageCSPad2x1<uint16_t>(&m_data[sect * m_sizeOf2x1Img]);  
            //image_2x1 -> printImage();

	    uint32_t rot_index = (int)(rotation/90);
            size_t ncols = image_2x1 -> getNCols(rot_index);
            size_t nrows = image_2x1 -> getNRows(rot_index);

	    uint32_t ixmin = (uint32_t)(xcenter - 0.5*nrows);
	    uint32_t iymin = (uint32_t)(ycenter - 0.5*ncols);

	    /*
            cout << "    sect:"     << sect
            	 << "    xcenter:"  << xcenter
            	 << "    ycenter:"  << ycenter
            	 << "    zcenter:"  << zcenter
            	 << "    ncols:"    << ncols
            	 << "    nrows:"    << nrows
            	 << "    rotation:" << rotation 
            	 << "    rot_index:"<< rot_index
            	 << "    ixmin:"    << ixmin
            	 << "    iymin:"    << iymin
                 << endl;
	    */


	    for (uint32_t row=0; row<nrows; row++) {
	    for (uint32_t col=0; col<ncols; col++) {

               m_quad_image[ixmin+row][iymin+col] = image_2x1 -> rotN90(row, col, rot_index);

	    }
	    }
}

//----------------

template <typename T>
void ImageCSPadQuad<T>::saveQuadImageInFile()
{
  uint32_t quad = m_quadpars -> getQuadNumber();
  string fname = "image_q";
         fname += boost::lexical_cast<string>( quad );
         fname += ".txt";

              m_quad_image_2d -> saveImageInFile(fname,0);
}

//----------------


//--------------
// Destructor --
//--------------
template <typename T>
ImageCSPadQuad<T>::~ImageCSPadQuad ()
{
  delete [] m_data; 
}

//-----------------------------------
// Instatiation of templated classes
//-----------------------------------

template class CSPadImage::ImageCSPadQuad<uint16_t>;

} // namespace CSPadImage
