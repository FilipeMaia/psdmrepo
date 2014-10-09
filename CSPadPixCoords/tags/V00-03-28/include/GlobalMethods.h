#ifndef CSPADPIXCOORDS_GLOBALMETHODS_H
#define CSPADPIXCOORDS_GLOBALMETHODS_H

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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace CSPadPixCoords {

/// @addtogroup CSPadPixCoords

/**
 *  @ingroup CSPadPixCoords
 *
 *  @brief Global methods for CSPadPixCoords package
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

using namespace std;


enum DATA_TYPE {ASDATA, FLOAT, DOUBLE, SHORT, UNSIGNED, INT, INT16, INT32, UINT, UINT8, UINT16, UINT32};

enum FILE_MODE {BINARY, TEXT, TIFF, PNG, METADTEXT};


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
  std::string stringTimeStamp(PSEvt::Event& evt, std::string fmt="%Y%m%d-%H%M%S%f"); //("%Y-%m-%d %H:%M:%S%f%z");
  std::string stringRunNumber(PSEvt::Event& evt, unsigned width=4);
  int getRunNumber(PSEvt::Event& evt);
  double doubleTime(PSEvt::Event& evt);
  unsigned fiducials(PSEvt::Event& evt);                  // returns 17-bits (131071) integer value: fiducials clock runs at 360Hz.
  unsigned eventCounterSinceConfigure(PSEvt::Event& evt); // returns 15-bits (32767)  integer value: event counter since Configure.
  void printSizeOfTypes();
  /// Define the shape or throw message that can not do that.
  void defineImageShape(PSEvt::Event& evt, const PSEvt::Source& m_str_src, const std::string& m_key, unsigned* shape);
  void printTimeStamp(PSEvt::Event& evt, int counter);

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
// Define inage shape in the event for specified type, str_src, and str_key 
  template <typename T, unsigned NDIM>
  std::string str_shape( const ndarray<const T,NDIM>& nda ) 
  {
    std::stringstream ss;
    const unsigned* shape = nda.shape();
    for (unsigned i=0; i<NDIM; ++i)
      ss << shape[i] << " ";
    return ss.str();
  } 

//--------------------
// Define inage shape in the event for specified type, str_src, and str_key 
  template <typename T>
  bool defineImageShapeForType(PSEvt::Event& evt, const PSEvt::Source& str_src, const std::string& str_key, unsigned* shape)
  {
    boost::shared_ptr< ndarray<const T,2> > img = evt.get(str_src, str_key);
    if (img.get()) {
      for(int i=0;i<2;i++) shape[i]=img->shape()[i];
      //shape=img->shape();
      return true;
    } 
    return false;
  }

//--------------------
// Save 2-D array in environment
  template <typename T>
  void save2DArrayInEnv(PSEnv::Env& env, const Pds::Src& src, const ndarray<const T,2>& ndarr)
  {
    boost::shared_ptr< ndarray<const T,2> > shp( new ndarray<const T,2>(ndarr) );
    env.calibStore().put(shp, src);
  }

//--------------------
// Save 2-D array in event
  template <typename T>
  void save2DArrayInEvent(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, const ndarray<const T,2>& ndarr)
  {
    boost::shared_ptr< ndarray<const T,2> > shp( new ndarray<const T,2>(ndarr) );
    evt.put(shp, src, key);
  }

// or its alias
/*
  template <typename T>
  void save2DArrInEvent(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, const ndarray<const T,2>& ndarr)
  {
    save2DArrayInEvent<T>(evt, src, key, ndarr);
  }
*/
//-------------------
  /**
   * @brief Save 3-D array in event, for src and key.
   * 
   * @param[in]  evt
   * @param[in]  src
   * @param[in]  key
   * @param[out] ndarr
   */

  template <typename T>
  void save3DArrInEvent(PSEvt::Event& evt, const Pds::Src& src, const std::string& key, const ndarray<const T,3>& ndarr)
  {
      boost::shared_ptr< ndarray<const T,3> > shp( new ndarray<const T,3>(ndarr) );
      evt.put(shp, src, key);
  }

//--------------------
/// Save ndarray in file
  template <typename T>
  void save2DArrayInFile(const std::string& fname, const ndarray<const T,2>& ndarr, bool print_msg, FILE_MODE file_type=TEXT)
  {  
    if (fname.empty()) {
      MsgLog("GlobalMethods", warning, "The output file name is empty. 2-d array is not saved.");
      return;
    }

    if( print_msg ) MsgLog("GlobalMethods", info, "Save 2-d array in file " << fname.c_str() << " file type:" << strOfDataTypeAndSize<T>());

    const unsigned* shape = ndarr.shape();
    const unsigned rows = shape[0]; 
    const unsigned cols = shape[1];
    const T* arr = ndarr.data();

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
//--------------------

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_GLOBALMETHODS_H
