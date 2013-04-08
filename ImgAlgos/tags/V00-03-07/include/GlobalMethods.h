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
  void save2DArrayInFile(const std::string& fname, const T* arr, const unsigned& rows, const unsigned& cols, bool print_msg )
  {  
    if (fname.empty()) {
      MsgLog("GlobalMethods", warning, "The output file name is empty. 2-d array is not saved.");
      return;
    }

    if( print_msg ) MsgLog("GlobalMethods", info, "Save 2-d array in file " << fname.c_str());
    std::ofstream out(fname.c_str());
          for (unsigned r = 0; r != rows; ++r) {
            for (unsigned c = 0; c != cols; ++c) {
              out << arr[r*cols + c] << ' ';
            }
            out << '\n';
          }
    out.close();
  }

//--------------------
// Save 2-D array in file
  template <typename T>
  void save2DArrayInFile(const std::string& fname, const boost::shared_ptr< ndarray<T,2> >& p_ndarr, bool print_msg )
  {  
    save2DArrayInFile<T> (fname, p_ndarr->data(), p_ndarr->shape()[0], p_ndarr->shape()[1], print_msg);
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