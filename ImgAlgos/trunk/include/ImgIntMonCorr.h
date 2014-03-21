#ifndef IMGALGOS_IMGINTMONCORR_H
#define IMGALGOS_IMGINTMONCORR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgIntMonCorr.
//      Apply corrections to 2d image using pedestals, background, gain factor, and mask.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <vector>
#include <fstream>  // for ostream, ofstream
#include <iostream> // for cout, puts etc.


//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ImgAlgos/ImgParametersV1.h"
#include "ImgAlgos/GlobalMethods.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *  Apply corrections to 2d image using pedestals, background, gain factor, and mask.
 *
 *  @ingroup ImgAlgos
 *
 *  @brief Example module class for psana
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

//-------------------

struct IntMonConfig {
  Source      source;
  std::string src_name;
  std::string name;
  unsigned    ch1;
  unsigned    ch2;
  unsigned    ch3;
  unsigned    ch4;
  unsigned    norm;
  unsigned    sele;
  double      imin;
  double      imax;
  double      iave;
};

//-------------------

struct Quartet{
  float v1;
  float v2; 
  float v3;
  float v4; 
   
  Quartet(float p1, float p2, float p3, float p4): v1(p1), v2(p2), v3(p3), v4(p4) {}
};

//-------------------

class ImgIntMonCorr : public Module {
public:

  // Default constructor
  ImgIntMonCorr (const std::string& name) ;

  // Destructor
  virtual ~ImgIntMonCorr () ;

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

  void printInputParameters();
  void printEventRecord(Event& evt);
  void printNormFactor();
  void readIntMonConfigFile();
  void printIntMonConfig();
  void printIntMonData(Event& evt, Env& env);
  bool procIntMonData(Event& evt, Env& env);
  Quartet getIntMonDataForSource(Event& evt, Env& env, const Source& src);
  Quartet getIntMonDataForSourceV1(Event& evt, Env& env, const Source& src);

protected:
  void init(Event& evt, Env& env);
  void procEvent(Event& evt, Env& env);
  //void saveImageInEvent(Event& evt);

private:

  Pds::Src        m_src;              // source address of the data object
  Source          m_str_src;          // string with source name
  std::string     m_key_in;           // string with key for input data
  std::string     m_key_out;          // string with key for output image
  std::string     m_fname_imon_cfg;   // string file name for intensity monitors configuration 
  unsigned        m_print_bits;       // bit mask for print options
  long            m_count;            // local event counter

  unsigned        m_shape[2];         // image shape
  unsigned        m_cols;             // number of columns in the image 
  unsigned        m_rows;             // number of rows    in the image 
  unsigned        m_size;             // image size = m_cols * m_rows (number of elements)

  bool            m_do_sele;          // flag: true = do selection
  bool            m_do_norm;          // flag: true = do normalization

  double          m_norm_factor;      // Normalization factor from intensity monitor

  std::vector<IntMonConfig> v_imon_cfg; // Vector filled from input file with parameters of intensity monitors configuration

//-------------------

    template <typename T>
    bool procEventForType(Event& evt)
    {
     	shared_ptr< ndarray<const T,2> > img = evt.get(m_str_src, m_key_in, &m_src);
     	if (img.get()) {

          if( m_print_bits & 1 && !m_count ) MsgLog( name(), info, " I/O data type: " << strOfDataTypeAndSize<T>() );

          if (! m_do_norm) {
             save2DArrayInEvent<T> (evt, m_src, m_key_out, *img.get());
             if( m_print_bits & 8 || m_print_bits & 16 ) 
               MsgLog( name(), info, stringOf2DArrayData<T>(*img.get(), std::string("Norm OFF - data in/out :")) );
       	     return true;
	  }

     	  const T* _rdat = img->data();

	  ndarray<T,2> cdata(m_shape);
	  T* _cdat  = cdata.data();
     	  for(unsigned i=0; i<m_size; i++) _cdat[i] = (T) (m_norm_factor * _rdat[i]);

          save2DArrayInEvent<T> (evt, m_src, m_key_out, cdata);

          if( m_print_bits & 8  ) MsgLog( name(), info, stringOf2DArrayData<T>(*img.get(), std::string("data in :")) );
          if( m_print_bits & 16 ) MsgLog( name(), info, stringOf2DArrayData<T>(cdata, std::string("data out:")) );
 
     	  return true;
     	} 
        return false;
    }  

//-------------------

};

} // namespace ImgAlgos

#endif // IMGALGOS_IMGINTMONCORR_H
