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
#include <math.h>
//#include <stdio.h>

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
//#define png_infopp_NULL (png_infopp)NULL
//#define int_p_NULL (int*)NULL

//#include <boost/mpl/vector.hpp>
#include <boost/gil/gil_all.hpp>

//#include <boost/mpl/vector.hpp>
//#include <boost/gil/typedefs.hpp>
//#include <boost/gil/extension/dynamic_image/any_image.hpp>
//#include <boost/gil/planar_pixel_reference.hpp>
//#include <boost/gil/color_convert.hpp>
//#include <boost/gil/typedefs.hpp>
//#include <boost/gil/image.hpp>
//#include <boost/gil/image_view.hpp>
//#include <boost/gil/image_view_factory.hpp>

#ifndef BOOST_GIL_NO_IO
#include <boost/gil/extension/io/png_io.hpp> 
#include <boost/gil/extension/io/tiff_io.hpp> 
//#include <boost/gil/extension/io/jpeg_io.hpp>

#include <boost/gil/extension/io/png_dynamic_io.hpp>
#include <boost/gil/extension/io/tiff_dynamic_io.hpp> 
//#include <boost/gil/extension/io/jpeg_dynamic_io.hpp> 
#endif

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/xtc/Src.hh"     // for srcToString( const Pds::Src& src )

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

using namespace boost::gil;
using namespace std;

//typedef boost::mpl::vector<gray8_image_t, gray16_image_t, gray32_image_t> my_images_t;

//typedef pixel<float,gray_layout_t>      gray_float_pixel_t;
//typedef image<gray_float_pixel_t,false> gray_float_image_t;
//typedef gray_float_image_t::view_t      gray_float_view_t; 

//typedef pixel<double,gray_layout_t>      gray_double_pixel_t;
//typedef image<gray_double_pixel_t,false> gray_double_image_t;
//typedef gray_double_image_t::view_t      gray_double_view_t; 

enum DATA_TYPE {ASDATA, FLOAT, DOUBLE, SHORT, UNSIGNED, INT, INT16, INT32, UINT, UINT8, UINT16, UINT32};

enum FILE_MODE {BINARY, TEXT, TIFF, PNG};

enum DETECTOR_TYPE {OTHER, CSPAD, CSPAD2X2};

const static int UnknownCM = -10000; 

class NDArrPars {
public:
  NDArrPars();
  NDArrPars(const unsigned ndim, const unsigned size, const unsigned* shape, const DATA_TYPE dtype, const Pds::Src& src);
  virtual ~NDArrPars(){}

  void setPars(const unsigned ndim, const unsigned size, const unsigned* shape, const DATA_TYPE dtype, const Pds::Src& src);
  void print();
  bool      is_set()    {return m_is_set;}
  unsigned  ndim()      {return m_ndim;}
  unsigned  size()      {return m_size;}
  unsigned* shape()     {return &m_shape[0];}
  DATA_TYPE dtype()     {return m_dtype;}
  const Pds::Src& src() {return m_src;}

private:
  unsigned  m_ndim;
  unsigned  m_size;
  unsigned  m_shape[5];
  DATA_TYPE m_dtype;
  Pds::Src  m_src;
  bool      m_is_set;

  // Copy constructor and assignment are disabled by default
  NDArrPars ( const NDArrPars& ) ;
  NDArrPars& operator = ( const NDArrPars& ) ;
};


class GlobalMethods  {
public:
  GlobalMethods() {}
  virtual ~GlobalMethods() {}

private:
  // Copy constructor and assignment are disabled by default
  GlobalMethods ( const GlobalMethods& ) ;
  GlobalMethods& operator = ( const GlobalMethods& ) ;
};

//--------------------

  std::string stringFromUint(unsigned number, unsigned width=6, char fillchar='0');
  std::string stringTimeStamp(PSEvt::Event& evt, std::string fmt="%Y%m%d-%H%M%S%f"); //("%Y-%m-%d %H:%M:%S%f%z");
  std::string stringRunNumber(PSEvt::Event& evt, unsigned width=4);
  int getRunNumber(PSEvt::Event& evt);
  double doubleTime(PSEvt::Event& evt);
  unsigned fiducials(PSEvt::Event& evt);                  // returns 17-bits (131071) integer value: fiducials clock runs at 360Hz.
  unsigned eventCounterSinceConfigure(PSEvt::Event& evt); // returns 15-bits (32767)  integer value: event counter since Configure.
  void printSizeOfTypes();
  /// Define the shape or throw message that can not do that.
  bool defineImageShape(PSEvt::Event& evt, const PSEvt::Source& src, const std::string& key, unsigned* shape);
  bool defineNDArrPars(PSEvt::Event& evt, const PSEvt::Source& src, const std::string& key, NDArrPars* ndarr_pars);
  void saveTextInFile(const std::string& fname, const std::string& text, bool print_msg);
  std::string stringInstrument(PSEnv::Env& env);
  std::string stringExperiment(PSEnv::Env& env);
  unsigned expNum(PSEnv::Env& env);
  std::string stringExpNum(PSEnv::Env& env, unsigned width=4);
  bool file_exists(std::string& fname);
  std::string srcToString( const Pds::Src& src ); // convert source address to string

//--------------------
//--------------------
//--------------------
//--------------------

//--------------------
/// For type=T returns the string with symbolic data type and its size, i.e. "d of size 8"
  template <typename T>
  std::string strOfDataTypeAndSize()
  {
    std::stringstream ss; ss << typeid(T).name() << " of size " << sizeof(T);
    return ss.str();
  }

//--------------------

  template <typename T>
    bool isSupportedDataType()
    {
	std::cout <<  "Input data type: " << strOfDataTypeAndSize<T>() << std::endl;
        if ( *typeid(T).name() != 't') {
	  cout <<  "Sorry, but saving images in PNG works for uint16_t data only..." << endl;
	  return false;
        }
	return true;
    }

//--------------------
/// Define inage shape in the event for specified type, src, and key 
  template <typename T>
  bool defineImageShapeForType(PSEvt::Event& evt, const PSEvt::Source& src, const std::string& key, unsigned* shape)
  {
    boost::shared_ptr< ndarray<T,2> > img = evt.get(src, key);
    if (img.get()) {
      for(int i=0;i<2;i++) shape[i]=img->shape()[i];
      //shape=img->shape();
      return true;
    } 
    return false;
  }

//--------------------
/// Define ndarray parameters in the event for specified type, src, and key 
  template <typename T>
  bool defineNDArrParsForType(PSEvt::Event& evt, const PSEvt::Source& src, const std::string& key, DATA_TYPE dtype, NDArrPars* ndarr_pars)
  {
    Pds::Src pds_src;    

    boost::shared_ptr< ndarray<T,2> > shp2 = evt.get(src, key, &pds_src);
    if (shp2.get()) { ndarr_pars->setPars(2, shp2->size(), shp2->shape(), dtype, pds_src); return true; } 

    boost::shared_ptr< ndarray<T,3> > shp3 = evt.get(src, key, &pds_src);
    if (shp3.get()) { ndarr_pars->setPars(3, shp3->size(), shp3->shape(), dtype, pds_src); return true; } 

    boost::shared_ptr< ndarray<T,4> > shp4 = evt.get(src, key, &pds_src);
    if (shp4.get()) { ndarr_pars->setPars(4, shp4->size(), shp4->shape(), dtype, pds_src); return true; } 

    boost::shared_ptr< ndarray<T,5> > shp5 = evt.get(src, key, &pds_src);
    if (shp5.get()) { ndarr_pars->setPars(5, shp5->size(), shp5->shape(), dtype, pds_src); return true; } 

    boost::shared_ptr< ndarray<T,1> > shp1 = evt.get(src, key, &pds_src);
    if (shp1.get()) { ndarr_pars->setPars(1, shp1->size(), shp1->shape(), dtype, pds_src); return true; } 

    return false;
  }

//--------------------
/// Save 2-D array in file
  template <typename T>
    bool save2DArrayInPNGForType(const std::string& fname, const T* arr, const unsigned& rows, const unsigned& cols)
    {
        using namespace boost::gil;

	//unsigned size = cols*rows;

	if ( *typeid(T).name() == *typeid(uint16_t).name() ) {
	  uint16_t* p_arr = (uint16_t*)&arr[0];
          gray16c_view_t image = interleaved_view(cols, rows, (const gray16_pixel_t*)p_arr, cols*sizeof(T));
          png_write_view(fname, image);
	  return true;
	}

	else if ( *typeid(T).name() == *typeid(uint8_t).name() ) {
	  uint8_t* p_arr = (uint8_t*)&arr[0];
          gray8c_view_t image = interleaved_view(cols, rows, (const gray8_pixel_t*)p_arr, cols*sizeof(T));
          png_write_view(fname, image);
	  return true;
	}

	return false;
    }


//--------------------
/// Save 2-D array in TIFF file
  template <typename T>
    bool save2DArrayInTIFFForType(const std::string& fname, const T* arr, const unsigned& rows, const unsigned& cols)
    {
        using namespace boost::gil;

	unsigned size = cols*rows;

	if ( *typeid(T).name() == *typeid(double).name() ) {
	  float* arr32f = new float[size]; 
          for (unsigned i=0; i<size; i++) { arr32f[i] = (float)arr[i]; }
	  gray32f_view_t image = interleaved_view(cols, rows, (gray32f_pixel_t*)&arr32f[0], cols*sizeof(float));
          tiff_write_view(fname, image);
	  return true;
	}

	else if ( *typeid(T).name() == *typeid(float).name() ) {
	  float* p_arr = (float*)&arr[0];
	  gray32f_view_t image = interleaved_view(cols, rows, (gray32f_pixel_t*)p_arr, cols*sizeof(T));
          tiff_write_view(fname, image);
	  return true;
	}

	else if ( *typeid(T).name() == *typeid(int).name() ) {
	  //int* p_arr = (int*)&arr[0];
	  //gray32c_view_t image = interleaved_view(cols, rows, reinterpret_cast<const gray32_pixel_t*>(p_arr), cols*sizeof(T));
	  float* arr32f = new float[size]; 
          for (unsigned i=0; i<size; i++) { arr32f[i] = (int)arr[i]; }
	  gray32f_view_t image = interleaved_view(cols, rows, (gray32f_pixel_t*)&arr32f[0], cols*sizeof(float));
          tiff_write_view(fname, image);
	  return true;
	}

	else if ( *typeid(T).name() == *typeid(uint16_t).name() ) {
	  uint16_t* p_arr = (uint16_t*)&arr[0];
          gray16c_view_t image = interleaved_view(cols, rows, (const gray16_pixel_t*)p_arr, cols*sizeof(T));
          tiff_write_view(fname, image);
	  return true;
	}

	else if ( *typeid(T).name() == *typeid(int16_t).name() ) {
	  int16_t* p_arr = (int16_t*)&arr[0];
          gray16c_view_t image = interleaved_view(cols, rows, (const gray16_pixel_t*)p_arr, cols*sizeof(T));
          tiff_write_view(fname, image);
	  return true;
	}

	else if ( *typeid(T).name() == *typeid(uint8_t).name() ) {
	  uint8_t* p_arr = (uint8_t*)&arr[0];
          gray8c_view_t image = interleaved_view(cols, rows, (const gray8_pixel_t*)p_arr, cols*sizeof(T));
          tiff_write_view(fname, image);
          png_write_view(fname+".png", image);
	  return true;
	}

	return false;
    }

//--------------------
/// Save 2-D array in file
  template <typename T>
  void save2DArrayInFile(const std::string& fname, const T* arr, const unsigned& rows, const unsigned& cols, bool print_msg, FILE_MODE file_type=TEXT)
  {  
    if (fname.empty()) {
      MsgLog("GlobalMethods", warning, "The output file name is empty. 2-d array is not saved.");
      return;
    }

    if( print_msg ) MsgLog("GlobalMethods", info, "Save 2-d array in file " << fname.c_str() << " file type:" << strOfDataTypeAndSize<T>());

    //======================
    if (file_type == TEXT) {
        std::ofstream out(fname.c_str()); 
	out << std::setprecision(9); // << std::setw(8) << std::setprecision(0) << std::fixed 
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

        //gray32f_image_t img(400,200);
        //gray32f_pixel_t col(150);
        //fill_pixels(view(img), col);
        //tiff_write_view("grayrect32.tiff", view(img));
        // DOES NOT WORK: png_write_view("grayrect32.png", view(img));
	//cout <<  "Save files : grayrect32.*" << endl;

        //png_write_view("grayrect32.png", view(img)); // DOES NOT WORK!!!

        //rgb8_image_t img(rows, cols);
        //rgb8_pixel_t red(255, 0, 0); fill_pixels(view(img), red);
        //png_write_view(fname, const_view(img));
        
        //type_from_x_iterator<T*>::view_t 
        //rgb8c_view_t image = interleaved_view(cols, rows, reinterpret_cast<const rgb8_pixel_t*>(&arr[0]), cols*sizeof(T));
        //rgb8c_view_t image = interleaved_view(cols, rows, (const rgb8_pixel_t*)arr, cols*sizeof(T));
        //rgb16c_view_t image = interleaved_view(cols/3, rows, (const rgb16_pixel_t*)arr, cols*sizeof(T));

	//if (! isSupportedDataType<T>()) return;
	//gray32c_view_t image = interleaved_view(cols, rows, reinterpret_cast<const gray32_pixel_t*>(&arr[0]), cols*sizeof(T));
	//gray16c_view_t image = interleaved_view(cols, rows, reinterpret_cast<const gray16_pixel_t*>(&arr[0]), cols*sizeof(T));
        //png_write_view(fname, image);

        if (save2DArrayInPNGForType<T>(fname, arr, rows, cols)) return;
        MsgLog("GlobalMethods", warning, "Input data type " << strOfDataTypeAndSize<T>() << " is not implemented for saving in PNG. File IS NOT saved!");
        return; 
    }

    //======================
    if (file_type == TIFF) {
        //MsgLog("GlobalMethods", warning, "Saving of images in TIFF format is not implemented yet.");

        if (save2DArrayInTIFFForType<T>(fname, arr, rows, cols)) return;
        MsgLog("GlobalMethods", warning, "Input data type " << strOfDataTypeAndSize<T>() << " is not implemented for saving in TIFF. File IS NOT saved!");
        return; 
    }

    //======================
  }


//--------------------
/// Save ndarray<T,2> in TEXT file
  template <typename T>
  void save2DArrayInFile(const std::string& fname, const ndarray<T,2>& p_ndarr, bool print_msg, FILE_MODE file_type=TEXT)
  {  
    save2DArrayInFile<T> (fname, p_ndarr.data(), p_ndarr.shape()[0], p_ndarr.shape()[1], print_msg, file_type);
  }

//--------------------
/// Save shared_ptr< ndarray<T,2> > in TEXT file
  template <typename T>
  void save2DArrayInFile(const std::string& fname, const boost::shared_ptr< ndarray<T,2> >& p_ndarr, bool print_msg, FILE_MODE file_type=TEXT)
  {  
    save2DArrayInFile<T> (fname, p_ndarr->data(), p_ndarr->shape()[0], p_ndarr->shape()[1], print_msg, file_type);
  }

//--------------------
/// Save 2-D array in file
  template <typename T>
  bool save2DArrayInFileForType(PSEvt::Event& evt, const PSEvt::Source& src, const std::string& key, const std::string& fname, bool print_msg, FILE_MODE file_type=TEXT)
  {
    boost::shared_ptr< ndarray<T,2> > img = evt.get(src, key);
    if ( ! img.get() ) return false; 
    save2DArrayInFile<T> (fname, img, print_msg, file_type);
    return true;
  }

//--------------------
/// Save 2-D array in event for type in case if key == "Image2D" 
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
/// Save 2-D array in event
  template <typename T>
  void save2DArrayInEvent(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, const ndarray<T,2>& data)
  {
    boost::shared_ptr< ndarray<T,2> > img2d( new ndarray<T,2>(data) );
    //boost::shared_ptr< ndarray<T,2> > img2d( &data );
    evt.put(img2d, src, key);
  }

// or its copy

  template <typename T>
  void save2DArrInEvent(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, const ndarray<T,2>& ndarr)
  {
    boost::shared_ptr< ndarray<T,2> > shp( new ndarray<T,2>(ndarr) );
    evt.put(shp, src, key);
  }

//-------------------
  /**
   * @brief Save 3-D array in event, for src and key.
   * 
   * @param[in]  evt
   * @param[in]  src
   * @param[in]  key
   * @param[in] ndarr
   */

  template <typename T>
  void save3DArrInEvent(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, const ndarray<T,3>& ndarr)
  {
      boost::shared_ptr< ndarray<T,3> > shp( new ndarray<T,3>(ndarr) );
      evt.put(shp, src, key);
  }

//-------------------
  /**
   * @brief Save N-D array in event, for src and key.
   * 
   * @param[in]  evt
   * @param[in]  src
   * @param[in]  key
   * @param[in]  arr
   * @param[in]  ndarr_pars
   * @param[in]  print_bits = 1-warnings
   */

  template <typename T>
  void saveNDArrInEvent(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, T* arr, NDArrPars* ndarr_pars, unsigned print_bits=1)
  {
      if ( print_bits & 1 && ! ndarr_pars->is_set() ) 
        MsgLog("saveNDArrInEvent", warning, "NDArrPars are not set for src: " << boost::lexical_cast<std::string>(src) << " key: " << key);

      unsigned ndim = ndarr_pars->ndim();
      if ( print_bits & 1 && ndim > 5 ) MsgLog("saveNDArrInEvent", warning, "ndim=" << ndim << " out of the range of implemented ndims [1,5]");
      if ( print_bits & 1 && ndim < 1 ) MsgLog("saveNDArrInEvent", warning, "ndim=" << ndim << " out of the range of implemented ndims [1,5]");

      if (ndim == 2) {
        boost::shared_ptr< ndarray<T,2> > shp( new ndarray<T,2>(arr, ndarr_pars->shape()) );
        evt.put(shp, src, key);
      } 

      else if (ndim == 3) {
        boost::shared_ptr< ndarray<T,3> > shp( new ndarray<T,3>(arr, ndarr_pars->shape()) );
        evt.put(shp, src, key);
      }

      else if (ndim == 4) {
        boost::shared_ptr< ndarray<T,4> > shp( new ndarray<T,4>(arr, ndarr_pars->shape()) );
        evt.put(shp, src, key);
      }

      else if (ndim == 5) {
        boost::shared_ptr< ndarray<T,5> > shp( new ndarray<T,5>(arr, ndarr_pars->shape()) );
        evt.put(shp, src, key);
      }

      else if (ndim == 1) {
        boost::shared_ptr< ndarray<T,1> > shp( new ndarray<T,1>(arr, ndarr_pars->shape()) );
        evt.put(shp, src, key);
      }
  }

//--------------------
/// Get string of the 2-D array partial data for test print purpose
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
/// String parameter s, consisting of values separated by space, is used as a stringstream to fill the output vector v. 
  template <typename T>
    void parse_string(std::string& s, std::vector<T>& v)
  {  
    std::stringstream ss(s);
    //cout << "parsing string: " << s << endl;
    T val;
    do { 
        ss >> val;
	v.push_back(val);
        //cout << val << endl;
    } while( ss.good() ); 
  }

//--------------------

// Load ndarray<T,2> from file TEXT fname
  template <typename T>
  void load2DArrayFromFile(const std::string& fname, const ndarray<T,2>& ndarr, bool print_msg=false, FILE_MODE file_type=TEXT)
  {  
    // std::ios_base::openmode mode = std::ios_base::out | std::ios_base::binary;
    // open file

    std::ifstream in(fname.c_str());   // or: (fname.c_str(), mode), where mode = std::ios::binary; 
      if (not in.good()) {
        const std::string msg = "Failed to open file: "+fname;
        MsgLogRoot(error, msg);
        throw std::runtime_error(msg);
      }

      if (print_msg) MsgLog("GlobalMethods", info, " load2DArrayFromFile: " << fname);
      
      // read all numbers
      T* it = ndarr.data();
      size_t count = 0;
      size_t size = ndarr.size();
      while(in and count != size) {
        in >> *it++;
        ++ count;
      }
      
      // check that we read whole array
      if (count != size) {
        const std::string msg = "File "+fname+" does not have enough data: ";
        MsgLogRoot(error, msg);
        throw std::runtime_error(msg);
      }
      
      // and no data left after we finished reading
      T tmp ;
      if ( in >> tmp ) {
        const std::string msg = "File "+fname
                              + " has extra data; read:" + stringFromUint(count,10,' ') 
                              + " expecting:"           + stringFromUint(size,10,' ');
        MsgLogRoot(error, msg);
        throw std::runtime_error(msg);
      }
      
      in.close();
  }


//--------------------
/// Save ndarray in file
  template <typename T>
  void saveNDArrayInFile(const std::string& fname, const T* arr, NDArrPars* ndarr_pars, bool print_msg, FILE_MODE file_type=TEXT)
  {  
    if (fname.empty()) {
      MsgLog("GlobalMethods", warning, "The output file name is empty. 2-d array is not saved.");
      return;
    }

    if( print_msg ) MsgLog("GlobalMethods", info, "Save 2-d array in file " << fname.c_str() << " file type:" << strOfDataTypeAndSize<T>());

    unsigned* shape = ndarr_pars->shape();
    unsigned  ndim  = ndarr_pars->ndim();

    unsigned cols = shape[ndim-1];
    unsigned rows = (ndim>1) ? ndarr_pars->size()/cols : 1; 

    //======================
    if (file_type == TEXT) {
        std::ofstream out(fname.c_str()); 
	out << std::setprecision(9); // << std::setw(8) << std::setprecision(0) << std::fixed 
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
  }



//--------------------
// averages all values in array "data" for length "length"
// that are below "threshold".  If the magniture of the correction
// is greater than "maxCorrection" leave "data" unchanged, otherwise
// subtract the average in place.  
// mask is either a null pointer (in which case nothing is masked)
// or a list of values arranged like the data where non-zero means ignore

   template <typename T>
   void commonMode(T* data, const uint16_t* mask, const unsigned length, const T threshold, const T maxCorrection, T& cm) {
     // do dumbest thing possible to start - switch to median
     // do 2nd dumbest thing possible to start - switch to median
     cm = 0;
     T* tmp = data;
     double sum = 0;
     int nSummed = 0;
     for (unsigned col=0; col<length; col++) {
       T cval = *tmp++;
       T mval = (mask) ? *mask++ : 0;
       if (mval==0 && cval<threshold) {
         nSummed++;
         sum += cval;
       }
     }
     
     if (nSummed>0) {
       T mean = (T)(sum/nSummed);
       if (fabs(mean)<=maxCorrection) {
         cm = mean;
         tmp = data;
         for (unsigned col=0; col<length; col++) {
   	*tmp++ -= cm;
         }
       }
     }
   }

//--------------------
// This method was originally designed by Andy for CSPAD in pdscalibdata/src/CsPadCommonModeSubV1.cpp
// - data type int16_t is changed to T
// - subtract CM inside this module

  /**
   *  Find common mode for an CsPad  section.
   *  
   *  Function will return UnknownCM value if the calculation 
   *  cannot be performed (or need not be performed).
   *  
   *  @param pars   array[3] of control parameters; mean_max, sigma_max, threshold on number of pixels/ADC count
   *  @param sdata  pixel data
   *  @param peddata  pedestal data, can be zero pointer
   *  @param pixStatus  pixel status data, can be zero pointer
   *  @param ssize  size of all above arrays
   *  @param stride increment for pixel indices
   */ 

   //float cm_corr =  findCommonMode(sdata, peddata, pixStatus, ssize, nSect); 

template <typename T>
float 
findCommonMode(const double* pars,
               T* sdata,
               //const int16_t* sdata,
               const float* peddata, 
               const uint16_t *pixStatus, 
               unsigned ssize,
               int stride = 1
               )
{
  // do we even need it
  //if (m_mode == None) return float(UnknownCM);

  // for now it does not make sense to calculate common mode
  // if pedestals are not known
  if (not peddata) return float(UnknownCM);
  
  // declare array for histogram
  const int low = -1000;
  const int high = 2000;
  const unsigned hsize = high-low;
  int hist[hsize];
  std::fill_n(hist, hsize, 0);
  unsigned long count = 0;
  
  // fill histogram
  for (unsigned c = 0, p = 0; c != ssize; ++ c, p += stride) {
    
    // ignore channels that re too noisy
    //if (pixStatus and (pixStatus[p] & 1)) continue;
    if (pixStatus and pixStatus[p]) continue; // Discard  pixels with any status > 0
    
    // pixel value with pedestal subtracted, rounded to integer
    double dval = sdata[p] - peddata[p];
    int val = dval < 0 ? int(dval - 0.5) : int(dval + 0.5);
    
    // histogram bin
    unsigned bin = unsigned(val - low);
    
    // increment bin value if in range
    if (bin < hsize) {
      ++hist[bin] ;
      ++ count;
    }      
  }

  MsgLog("findCommonMode", debug, "histo filled count = " << count);
  
  // analyze histogram now, first find peak position
  // as the position of the lowest bin with highest count 
  // larger than 100 and which has a bin somewhere on 
  // right side with count dropping by half
  int peakPos = -1;
  int peakCount = -1;
  int hmRight = hsize;
  int thresh = int(pars[2]);
  if(thresh<=0) thresh=100;
  for (unsigned i = 0; i < hsize; ++ i ) {
    if (hist[i] > peakCount and hist[i] > thresh) {
      peakPos = i;
      peakCount = hist[i];
    } else if (peakCount > 0 and hist[i] <= peakCount/2) {
      hmRight = i;
      break;
    }
  }

  // did we find anything resembling
  if (peakPos < 0) {
    MsgLog("findCommonMode", debug, "peakPos = " << peakPos);
    return float(UnknownCM);
  }

  // find half maximum channel on left side
  int hmLeft = -1;
  for (int i = peakPos; hmLeft < 0 and i >= 0; -- i) {
    if(hist[i] <= peakCount/2) hmLeft = i;
  }
  MsgLog("findCommonMode", debug, "peakPos = " << peakPos << " peakCount = " << peakCount 
      << " hmLeft = " << hmLeft << " hmRight = " << hmRight);
  
  // full width at half maximum
  int fwhm = hmRight - hmLeft;
  double sigma = fwhm / 2.36;

  // calculate mean and sigma
  double mean = peakPos;
  for (int j = 0; j < 2; ++j) {
    int s0 = 0;
    double s1 = 0;
    double s2 = 0;
    int d = int(sigma*2+0.5);
    for (int i = std::max(0,peakPos-d); i < int(hsize) and i <= peakPos+d; ++ i) {
      s0 += hist[i];
      s1 += (i-mean)*hist[i];
      s2 += (i-mean)*(i-mean)*hist[i];
    }
    mean = mean + s1/s0;
    sigma = std::sqrt(s2/s0 - (s1/s0)*(s1/s0));
  }
  mean += low;
  
  MsgLog("findCommonMode", debug, "mean = " << mean << " sigma = " << sigma);

  // limit the values to some reasonable numbers
  if (mean > pars[0] or sigma > pars[1]) return float(UnknownCM);
  
  //--------------------
  // subtract CM 
  for (unsigned c = 0, p = 0; c < ssize; ++ c, p += stride) {
        double val = sdata[p] - mean;
        sdata[p] = val < 0 ? int(val - 0.5) : int(val + 0.5);
  } 

  return mean;
}

//--------------------
//--------------------
//--------------------
//--------------------

} // namespace ImgAlgos

#endif // IMGALGOS_GLOBALMETHODS_H
