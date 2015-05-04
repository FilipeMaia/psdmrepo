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

#include "pdscalibdata/NDArrIOV1.h"

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

/**
 *  @defgroup ImgAlgos ImgAlgos package 
 *  @brief Package ImgAlgos contains a collection of psana modules and algorithms for LCLS data processing
 *
 *  Modules in this package can be destinguished be their names as follows. 
 *  @li Psana modules:
 *  \n  Acqiris* - work with acqiris data
 *  \n  CSPad* - work with CSPAD data
 *  \n  *detector-name* - work with specific detector data
 *  \n  *ImageProducer - produces 2-d ndarray from raw data and save it in the event store
 *  \n  NDArr* - modules for ndarray processing
 *  \n  Img* - modules for image (2-d ndarray) processing
 *  
 *  @li Algorithms:
 *  \n  CorAna* - modules (psana and standalone) developed for time-correlation analysis 
 *  \n  Geometry* - unified heirarchial detector geometry description
 *  \n  Global* - global methods
 *
 */


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

 enum DATA_TYPE {NONDEFDT, ASDATA, ASINP, FLOAT, DOUBLE, SHORT, UNSIGNED, INT, INT16, INT32, UINT, UINT8, UINT16, UINT32};

 enum FILE_MODE {BINARY, TEXT, TIFF, PNG, METADTEXT};

 enum DETECTOR_TYPE {OTHER, CSPAD, CSPAD2X2, PNCCD, PRINCETON, ACQIRIS, TM6740, 
                     OPAL1000, OPAL2000, OPAL4000, OPAL8000,
                     ANDOR, ORCAFL40, FCCD960, EPIX, EPIX100A, EPIX10K};

 // const static int UnknownCM = -10000; 

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

//--------------------

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
  bool defineNDArrPars(PSEvt::Event& evt, const PSEvt::Source& src, const std::string& key, NDArrPars* ndarr_pars, bool print_wng=false);
  void saveTextInFile(const std::string& fname, const std::string& text, bool print_msg);
  std::string stringInstrument(PSEnv::Env& env);
  std::string stringExperiment(PSEnv::Env& env);
  unsigned expNum(PSEnv::Env& env);
  std::string stringExpNum(PSEnv::Env& env, unsigned width=4);
  bool file_exists(std::string& fname);
  std::string srcToString( const Pds::Src& src ); // convert source address to string
  DETECTOR_TYPE detectorTypeForStringSource(const std::string& str_src);
  DETECTOR_TYPE detectorTypeForSource(PSEvt::Source& src);
  std::string calibGroupForDetType(const DETECTOR_TYPE det_type);
  std::string calibGroupForSource(PSEvt::Source& src);
  std::string split_string_left(const std::string& s, size_t& pos, const char& sep=':');
  std::string strDataType(const DATA_TYPE& dtype);

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

  template <typename T>
    DATA_TYPE dataType()
    {
      if      ( typeid(T) == typeid(double  )) return DOUBLE;
      else if ( typeid(T) == typeid(float   )) return FLOAT;
      else if ( typeid(T) == typeid(int     )) return INT;
      else if ( typeid(T) == typeid(int32_t )) return INT32;
      else if ( typeid(T) == typeid(uint32_t)) return UINT32;
      else if ( typeid(T) == typeid(uint16_t)) return UINT16;
      else if ( typeid(T) == typeid(uint8_t )) return UINT8;
      else if ( typeid(T) == typeid(int16_t )) return INT16;
      else if ( typeid(T) == typeid(short   )) return SHORT;
      else if ( typeid(T) == typeid(unsigned)) return UNSIGNED;
      else                                     return NONDEFDT;
    }

//--------------------

  template <typename T>
    std::string strDataType()
    {
      return strDataType(dataType<T>());
    }

//--------------------
/// Define inage shape in the event for specified type, src, and key 
  template <typename T>
  bool defineImageShapeForType(PSEvt::Event& evt, const PSEvt::Source& src, const std::string& key, unsigned* shape)
  {
    boost::shared_ptr< ndarray<const T,2> > img_const = evt.get(src, key);
    if (img_const.get()) {
      for(int i=0;i<2;i++) shape[i]=img_const->shape()[i];
      //shape=img->shape();
      return true;
    } 

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
  template <typename T, unsigned NDim>
  bool defineNDArrParsForTypeNDim(PSEvt::Event& evt, const PSEvt::Source& src, const std::string& key, DATA_TYPE dtype, NDArrPars* ndarr_pars)
  {
    Pds::Src pds_src;    

    boost::shared_ptr< ndarray<T,NDim> >  shp = evt.get(src, key, &pds_src);
    if (shp.get()) { ndarr_pars->setPars(NDim, shp->size(), shp->shape(), dtype, pds_src); return true; } 

    return false;
  }

//--------------------
/// Define ndarray parameters in the event for specified type, src, and key 
  template <typename T>
  bool defineNDArrParsForType(PSEvt::Event& evt, const PSEvt::Source& src, const std::string& key, DATA_TYPE dtype, NDArrPars* ndarr_pars)
  {
    // CONST
    if (defineNDArrParsForTypeNDim<const T,2>(evt, src, key, dtype, ndarr_pars)) return true;
    if (defineNDArrParsForTypeNDim<const T,3>(evt, src, key, dtype, ndarr_pars)) return true;
    if (defineNDArrParsForTypeNDim<const T,4>(evt, src, key, dtype, ndarr_pars)) return true;
    if (defineNDArrParsForTypeNDim<const T,5>(evt, src, key, dtype, ndarr_pars)) return true;
    if (defineNDArrParsForTypeNDim<const T,1>(evt, src, key, dtype, ndarr_pars)) return true;

    // NON-CONST
    if (defineNDArrParsForTypeNDim<T,2>(evt, src, key, dtype, ndarr_pars)) return true;
    if (defineNDArrParsForTypeNDim<T,3>(evt, src, key, dtype, ndarr_pars)) return true;
    if (defineNDArrParsForTypeNDim<T,4>(evt, src, key, dtype, ndarr_pars)) return true;
    if (defineNDArrParsForTypeNDim<T,5>(evt, src, key, dtype, ndarr_pars)) return true;
    if (defineNDArrParsForTypeNDim<T,1>(evt, src, key, dtype, ndarr_pars)) return true;

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
/// Save ndarray<T,2> in TEXT file NON-CONST T
//  template <typename T>
//  void save2DArrayInFile(const std::string& fname, const ndarray<T,2>& p_ndarr, bool print_msg, FILE_MODE file_type=TEXT)
//  {  
//    save2DArrayInFile<T> (fname, p_ndarr.data(), p_ndarr.shape()[0], p_ndarr.shape()[1], print_msg, file_type);
//  }

//--------------------
/// Save ndarray<const T,2> in TEXT file
  template <typename T>
  void save2DArrayInFile(const std::string& fname, const ndarray<const T,2>& p_ndarr, bool print_msg, FILE_MODE file_type=TEXT)
  {  
    save2DArrayInFile<T> (fname, p_ndarr.data(), p_ndarr.shape()[0], p_ndarr.shape()[1], print_msg, file_type);
  }

//--------------------
/// Save shared_ptr< ndarray<const T,2> > in TEXT file WITH NON-CONST
  template <typename T>
  void save2DArrayInFile(const std::string& fname, const boost::shared_ptr< ndarray<T,2> >& p_ndarr, bool print_msg, FILE_MODE file_type=TEXT)
  {  
    save2DArrayInFile<T> (fname, p_ndarr->data(), p_ndarr->shape()[0], p_ndarr->shape()[1], print_msg, file_type);
  }

//--------------------
/// Save shared_ptr< ndarray<T,2> > in TEXT file WITH CONST
  template <typename T>
  void save2DArrayInFile(const std::string& fname, const boost::shared_ptr< ndarray<const T,2> >& p_ndarr, bool print_msg, FILE_MODE file_type=TEXT)
  {  
    save2DArrayInFile<T> (fname, p_ndarr->data(), p_ndarr->shape()[0], p_ndarr->shape()[1], print_msg, file_type);
  }

//--------------------
/// Save 2-D array in file
  template <typename T>
  bool save2DArrayInFileForType(PSEvt::Event& evt, const PSEvt::Source& src, const std::string& key, const std::string& fname, bool print_msg, FILE_MODE file_type=TEXT)
  {
    boost::shared_ptr< ndarray<const T,2> > shp_const = evt.get(src, key);
    if ( shp_const.get() ) {
        save2DArrayInFile<T> (fname, shp_const, print_msg, file_type);
        return true;
    }

    boost::shared_ptr< ndarray<T,2> > shp = evt.get(src, key);
    if ( shp.get() ) {
        save2DArrayInFile<T> (fname, shp, print_msg, file_type);
        return true;
    }

    return false; 
  }

//--------------------
/// Save 2-D array in event for type in case if key == "Image2D" 
  template <typename T>
  bool saveImage2DInFileForType(PSEvt::Event& evt, const PSEvt::Source& src, const std::string& key, const std::string& fname, bool print_msg)
  {
    boost::shared_ptr< CSPadPixCoords::Image2D<T> > shp = evt.get(src, key);
    if ( ! shp.get() ) return false; 
    if( print_msg ) MsgLog("GlobalMethods::saveImage2DInFileForType", info, "Get image as Image2D<T> from event and save it in file");
    shp -> saveImageInFile(fname,0);
    return true;
  }

//--------------------
/// Save 1-D array in the calibStore for const T
  template <typename T>
  void save1DArrayInCalibStore(PSEnv::Env& env, const Pds::Src& src, const std::string& key, const ndarray<const T,1>& data)
  {
    boost::shared_ptr< ndarray<const T,1> > shp( new ndarray<const T,1>(data) );
    env.calibStore().put(shp, src, key);
  }

//--------------------
//--------------------
//--------------------
//--------------------
//--------------------
/// Save N-D array in event for const T
  template <typename T, unsigned NDim>
  void saveNDArrayInEvent(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, const ndarray<const T,NDim>& nda)
  {
    boost::shared_ptr< ndarray<const T,NDim> > shp( new ndarray<const T,NDim>(nda) );
    evt.put(shp, src, key);
  }


/// Save 1-D array in event for const T
  template <typename T>
  void save1DArrayInEvent(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, const ndarray<const T,1>& data)
  {
    boost::shared_ptr< ndarray<const T,1> > shp( new ndarray<const T,1>(data) );
    evt.put(shp, src, key);
  }

//--------------------
/// Save 2-D array in event for const T
  template <typename T>
  void save2DArrayInEvent(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, const ndarray<const T,2>& data)
  {
    boost::shared_ptr< ndarray<const T,2> > shp( new ndarray<const T,2>(data) );
    evt.put(shp, src, key);
  }


//--------------------
/// Save 3-D array in event for const T
  template <typename T>
  void save3DArrayInEvent(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, const ndarray<const T,3>& data)
  {
    boost::shared_ptr< ndarray<const T,3> > shp( new ndarray<const T,3>(data) );
    evt.put(shp, src, key);
  }

//--------------------
/// Save 4-D array in event for const T
  template <typename T>
  void save4DArrayInEvent(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, const ndarray<const T,4>& data)
  {
    boost::shared_ptr< ndarray<const T,4> > shp( new ndarray<const T,4>(data) );
    evt.put(shp, src, key);
  }

//--------------------
/// Save 5-D array in event for const T
  template <typename T>
  void save5DArrayInEvent(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, const ndarray<const T,5>& data)
  {
    boost::shared_ptr< ndarray<const T,5> > shp( new ndarray<const T,5>(data) );
    evt.put(shp, src, key);
  }

//--------------------
/// Save 2-D non-const T array in event
  template <typename T>
  void saveNonConst2DArrayInEvent(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, const ndarray<T,2>& data)
  {
    boost::shared_ptr< ndarray<T,2> > shp( new ndarray<T,2>(data) );
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
  void save3DArrInEvent(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, const ndarray<const T,3>& ndarr)
  {
      boost::shared_ptr< ndarray<const T,3> > shp( new ndarray<const T,3>(ndarr) );
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

  template <typename T, unsigned NDim>
  void saveNDArrInEventForTypeNDim(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, T* arr, NDArrPars* ndarr_pars, unsigned print_bits=1)
  {
      boost::shared_ptr< ndarray<T,NDim> > shp( new ndarray<T,NDim>(arr, ndarr_pars->shape()) );
      evt.put(shp, src, key);    
  }

//-------------------

  template <typename T>
  void saveNDArrInEvent(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, T* arr, NDArrPars* ndarr_pars, unsigned print_bits=1)
  {
      if ( print_bits & 1 && ! ndarr_pars->is_set() ) 
        MsgLog("saveNDArrInEvent", warning, "NDArrPars are not set for src: " << boost::lexical_cast<std::string>(src) << " key: " << key);

      unsigned ndim = ndarr_pars->ndim();
      if ( print_bits & 1 && ndim > 5 ) MsgLog("saveNDArrInEvent", warning, "ndim=" << ndim << " out of the range of implemented ndims [1,5]");
      if ( print_bits & 1 && ndim < 1 ) MsgLog("saveNDArrInEvent", warning, "ndim=" << ndim << " out of the range of implemented ndims [1,5]");

      if      (ndim == 2) saveNDArrInEventForTypeNDim<const T,2>(evt, src, key, arr, ndarr_pars, print_bits);
      else if (ndim == 3) saveNDArrInEventForTypeNDim<const T,3>(evt, src, key, arr, ndarr_pars, print_bits);
      else if (ndim == 4) saveNDArrInEventForTypeNDim<const T,4>(evt, src, key, arr, ndarr_pars, print_bits);
      else if (ndim == 5) saveNDArrInEventForTypeNDim<const T,5>(evt, src, key, arr, ndarr_pars, print_bits);
      else if (ndim == 1) saveNDArrInEventForTypeNDim<const T,1>(evt, src, key, arr, ndarr_pars, print_bits);
  }

//--------------------
/// Get string of the 2-D array partial data for test print purpose with "const T"
  template <typename T>
    std::string stringOf2DArrayData(const ndarray<const T,2>& data, std::string comment="",
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
/// Re-call to pdscalibdata::NDArrIOV1,
/// Save N-D array in file with metadata

  template <typename T, unsigned NDim>
    void saveNDArrayInFile(const ndarray<const T,NDim>& nda, const std::string& fname, const std::vector<std::string>& comments = std::vector<std::string>(), const unsigned& print_bits=1)
  {
    pdscalibdata::NDArrIOV1<T,NDim>::save_ndarray(nda, fname, comments, print_bits);
  }


//--------------------
/// Save ndarray in file
  template <typename T>
    void saveNDArrayInFile(const std::string& fname, const T* arr, NDArrPars* ndarr_pars, bool print_msg, FILE_MODE file_type=TEXT,
                           const std::vector<std::string>& comments = std::vector<std::string>())
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

    if (file_type == METADTEXT) {

      //std::vector<std::string> comments;
      //comments.push_back("PRODUCER   pdscalibdata/GlobalMethods/saveNDArrayInFile");

      if (ndim == 2) {
        ndarray<const T,2> nda(arr, shape);
        pdscalibdata::NDArrIOV1<T,2>::save_ndarray(nda, fname, comments);
      }

      else if (ndim == 3) {
        ndarray<const T,3> nda(arr, shape);
        pdscalibdata::NDArrIOV1<T,3>::save_ndarray(nda, fname, comments);
      }

      else if (ndim == 4) {
        ndarray<const T,4> nda(arr, shape);
        pdscalibdata::NDArrIOV1<T,4>::save_ndarray(nda, fname, comments);
      }

      else if (ndim == 5) {
        ndarray<const T,5> nda(arr, shape);
        pdscalibdata::NDArrIOV1<T,5>::save_ndarray(nda, fname, comments);
      }

      else if (ndim == 1) {
        ndarray<const T,1> nda(arr, shape);
        pdscalibdata::NDArrIOV1<T,1>::save_ndarray(nda, fname, comments);
      }

      else {
        if( print_msg ) MsgLog("pdscalibdata/GlobalMethods/saveNDArrayInFile", error, 
                               "ndarray of type " << strOfDataTypeAndSize<T>()
			       << " Ndim=" << ndim << " can not be saved in file: " << fname.c_str() ); 	
      }

      return;
    }

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
//--------------------
//--------------------
//--------------------

} // namespace ImgAlgos

#endif // IMGALGOS_GLOBALMETHODS_H
