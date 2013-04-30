#ifndef IMGALGOS_GLOBALMETHODS_H
#define IMGALGOS_GLOBALMETHODS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class GlobalMethods.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include <string>
#include <fstream>   // ofstream
#include <iomanip>   // for setw, setfill
#include <sstream>   // for stringstream
#include <iostream>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "PSEvt/Event.h"
#include "PSEnv/Env.h"
#include "PSEvt/Source.h"
#include "MsgLogger/MsgLogger.h"
#include "CSPadPixCoords/Image2D.h"

//For save in PNG and TIFF formats
#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL
#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/png_dynamic_io.hpp>
#include <boost/gil/extension/io/tiff_dynamic_io.hpp> 
//#include <boost/gil/extension/io/tiff_io.hpp> 
//#include <boost/gil/extension/io/dynamic_io.hpp>


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *  @ingroup ImgAlgos
 *
 *  @brief Global methods for ImgAlgos package
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

  enum FILE_MODE {BINARY, TEXT, TIFF, PNG};

//using namespace boost::gil;

class GlobalMethods  {
public:
  GlobalMethods () ;
  virtual ~GlobalMethods () ;

private:
  // Copy constructor and assignment are disabled by default
  GlobalMethods ( const GlobalMethods& ) ;
  GlobalMethods& operator = ( const GlobalMethods& ) ;
};

//--------------------

  std::string stringFromUint(unsigned number, unsigned width=6, char fillchar='0');
  std::string stringRunNumber(PSEvt::Event& evt, unsigned width=4);
  std::string stringTimeStamp(PSEvt::Event& evt, std::string fmt="%Y%m%d-%H%M%S%f"); //("%Y-%m-%d %H:%M:%S%f%z");
  double doubleTime(PSEvt::Event& evt);
  unsigned fiducials(PSEvt::Event& evt);                  // returns 17-bits (131071) integer value: fiducials clock runs at 360Hz.
  unsigned eventCounterSinceConfigure(PSEvt::Event& evt); // returns 15-bits (32767)  integer value: event counter since Configure.
  void printSizeOfTypes();
  /// Define the shape or throw message that can not do that.
  void defineImageShape(PSEvt::Event& evt, const PSEvt::Source& m_str_src, const std::string& m_key, unsigned* shape);

//--------------------
//--------------------
//--------------------
//--------------------

//--------------------
// For type=T returns the string with symbolic data type and its size, i.e. "d of size 8"
  template <typename T>
  std::string strOfDataTypeAndSize()
  {
    std::stringstream ss; ss << typeid(T).name() << " of size " << sizeof(T);
    return ss.str();
  }

//--------------------
// Define inage shape in the event for specified type, str_src, and str_key 
  template <typename T>
  bool defineImageShapeForType(PSEvt::Event& evt, const PSEvt::Source& str_src, const std::string& str_key, unsigned* shape)
  {
    boost::shared_ptr< ndarray<T,2> > img = evt.get(str_src, str_key);
    if (img.get()) {
      for(int i=0;i<2;i++) shape[i]=img->shape()[i];
      //shape=img->shape();
      return true;
    } 
    return false;
  }

//--------------------
// Save 2-D array in file
  template <typename T>
  void save2DArrayInFile(const std::string& fname, const T* arr, const unsigned& rows, const unsigned& cols, bool print_msg, FILE_MODE file_type=TEXT)
  {  
    if (fname.empty()) {
      MsgLog("GlobalMethods", warning, "The output file name is empty. 2-d array is not saved.");
      return;
    }

    if( print_msg ) MsgLog("GlobalMethods", info, "Save 2-d array in file " << fname.c_str() << " file type:" << file_type);

    //======================
    if (file_type == TEXT) {
        std::ofstream out(fname.c_str());
              for (unsigned r = 0; r != rows; ++r) {
                for (unsigned c = 0; c != cols; ++c) {
                  out << arr[r*cols + c] << ' ';
                }
                out << '\n';
              }
        out.close();
        return; 
    }

    //======================
    if (file_type == BINARY) {
        std::ios_base::openmode mode = std::ios_base::out | std::ios_base::binary;
        std::ofstream out(fname.c_str(), mode);
        //out.write(reinterpret_cast<const char*>(arr), rows*cols*sizeof(T));
        for (unsigned r = 0; r != rows; ++r) {
	  const T* data = &arr[r*cols];
          out.write(reinterpret_cast<const char*>(data), cols*sizeof(T));
	}
        out.close();
        return; 
    }

    //======================
    if (file_type == PNG) {
        using namespace boost::gil;
        //rgb8_image_t img(rows, cols);
        //rgb8_pixel_t red(255, 0, 0); fill_pixels(view(img), red);
        //png_write_view(fname, const_view(img));
        
        //type_from_x_iterator<T*>::view_t 
        //rgb8c_view_t image = interleaved_view(cols, rows, reinterpret_cast<const rgb8_pixel_t*>(&arr[0]), cols*sizeof(T));
        //rgb8c_view_t image = interleaved_view(cols, rows, (const rgb8_pixel_t*)arr, cols*sizeof(T));
        //rgb16c_view_t image = interleaved_view(cols/3, rows, (const rgb16_pixel_t*)arr, cols*sizeof(T));
        gray16c_view_t image = interleaved_view(cols, rows, reinterpret_cast<const gray16_pixel_t*>(&arr[0]), cols*sizeof(T));
        png_write_view(fname, image);
        return; 
    }

    //======================
    if (file_type == TIFF) {
        //MsgLog("GlobalMethods", warning, "Saving of images in TIFF format is not implemented yet.");
        using namespace boost::gil;
        gray16c_view_t image = interleaved_view(cols, rows, reinterpret_cast<const gray16_pixel_t*>(&arr[0]), cols*sizeof(T));
        tiff_write_view(fname, image);
        return; 
    }

    //======================
  }

//--------------------
// Save 2-D array in file
  template <typename T>
  void save2DArrayInFile(const std::string& fname, const boost::shared_ptr< ndarray<T,2> >& p_ndarr, bool print_msg, FILE_MODE file_type=TEXT)
  {  
    save2DArrayInFile<T> (fname, p_ndarr->data(), p_ndarr->shape()[0], p_ndarr->shape()[1], print_msg, file_type);
  }

//--------------------
// Save 2-D array in event for type
  template <typename T>
  bool save2DArrayInFileForType(PSEvt::Event& evt, const PSEvt::Source& src, const std::string& key, const std::string& fname, bool print_msg, FILE_MODE file_type=TEXT)
  {
    boost::shared_ptr< ndarray<T,2> > img = evt.get(src, key);
    if ( ! img.get() ) return false; 
    save2DArrayInFile<T> (fname, img, print_msg, file_type);
    return true;
  }

//--------------------
// Save 2-D array in event for type in case if key == "Image2D" 
  template <typename T>
  bool saveImage2DInFileForType(PSEvt::Event& evt, const PSEvt::Source& src, const std::string& key, const std::string& fname, bool print_msg)
  {
    boost::shared_ptr< CSPadPixCoords::Image2D<T> > img2d = evt.get(src, key);
    if ( ! img2d.get() ) return false; 
    if( print_msg ) MsgLog("GlobalMethods::saveImage2DInFileForType", info, "Get image as Image2D<T> from event and save it in file");
    img2d -> saveImageInFile(fname,0);
    return true;
  }

//--------------------
// Save 2-D array in event
  template <typename T>
  void save2DArrayInEvent(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, const ndarray<T,2>& data)
  {
    boost::shared_ptr< ndarray<T,2> > img2d( new ndarray<T,2>(data) );
    evt.put(img2d, src, key);
  }

//--------------------
// Get string of the 2-D array partial data for test print purpose
  template <typename T>
    std::string stringOf2DArrayData(const ndarray<T,2>& data, std::string comment="",
                                  unsigned row_min=0, unsigned row_max=1, 
                                  unsigned col_min=0, unsigned col_max=10 )
  {
      std::stringstream ss;
      ss << comment << std::setprecision(3); 
          for (unsigned r = row_min; r < row_max; ++ r) {
            for (unsigned c = col_min; c < col_max; ++ c ) ss << " " << std::setw(7) << data[r][c]; 
            if(row_max > 1) ss << " ...\n";
	  }
      return ss.str();
  }

//--------------------
//--------------------
//--------------------
//--------------------
//--------------------

} // namespace ImgAlgos

#endif // IMGALGOS_GLOBALMETHODS_H
