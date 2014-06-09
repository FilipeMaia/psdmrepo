#ifndef IMGALGOS_IMGINTFORBINS_H
#define IMGALGOS_IMGINTFORBINS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgIntForBins.
//      Evaluates average 2d image intensity in bins defined by the map,
//      saves output file with I(bin,event).
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <fstream> // for std::ofstream operator << 
#include <sstream> // for stringstream

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
 *  Evaluates average 2d image intensity in bins defined by the map,
 *  saves output file with I(bin,event).
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

class ImgIntForBins : public Module {
public:

  typedef double data_out_t;

  // Default constructor
  ImgIntForBins (const std::string& name) ;

  // Destructor
  virtual ~ImgIntForBins () ;

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

protected:
  void init(Event& evt, Env& env);
  void procEvent(Event& evt, Env& env);
  //void normBkgd();
  void saveIntensityBinsInEvent(Event& evt);
  std::string strRecord();

private:

  Pds::Src        m_src;              // source address of the data object
  Source          m_str_src;          // string with source name
  std::string     m_key_in;           // string with key for input data
  std::string     m_key_out;          // string with key for output image
  std::string     m_fname_map_bins;   // string file name for input map of image size with bin indexes per pix  
  std::string     m_fname_int_bins;   // string file name for output intensity array over bins vs event 
  unsigned        m_nbins;            // bit mask for print options
  unsigned        m_print_bits;       // bit mask for print options
  long            m_count;            // local event counter

  bool            m_do_binning;       // logic on/off procedure

  unsigned        m_shape[2];         // image shape
  unsigned        m_cols;             // number of columns in the image 
  unsigned        m_rows;             // number of rows    in the image 
  unsigned        m_size;             // image size = m_cols * m_rows (number of elements)

  ImgParametersV1* m_map_bins;

  unsigned*       m_inds;

  unsigned*       m_sum_stat;
  data_out_t*     m_sum_intens; 
  data_out_t*     m_intens_ave;

  std::ofstream p_out;

//-------------------

  template <typename T>
    bool procEventForType(Event& evt)
    {
     	shared_ptr< ndarray<const T,2> > img = evt.get(m_str_src, m_key_in, &m_src);
     	if (img.get()) {

     	  const T* _rdat = img->data();


          std::fill_n(m_sum_intens, int(m_nbins), data_out_t(0));    
          std::fill_n(m_sum_stat,   int(m_nbins), unsigned(0));    

	  unsigned bin(0);
     	  for(unsigned pix=0; pix<m_size; pix++) {

	    bin = m_inds[pix];
	    if( !(bin < m_nbins) ) continue;

	      m_sum_intens[bin] += (data_out_t)_rdat[pix];
              m_sum_stat  [bin] ++;
	  }


     	  for(unsigned bin=0; bin<m_nbins; bin++) {

	    m_intens_ave[bin] = (m_sum_stat[bin]<1) ? 0 : m_sum_intens[bin] / m_sum_stat[bin];

	  }

          std::string s = strRecord();
          p_out.write(s.c_str(), s.size());

          // std::cout  << s.size() << " " << s;
 
	  //ndarray<TOUT,2> cdata(m_shape);
	  //TOUT* p_cdata = cdata.data();

          if( m_print_bits &  8 ) MsgLog( name(), info, s.substr(0,100) + "..." );
          //if( m_print_bits & 16 ) MsgLog( name(), info, stringOf2DArrayData<TOUT>(cdata, std::string("Calibr. data:")) );
 	  
          //save2DArrayInEvent<TOUT> (evt, m_src, m_key_out, cdata);

     	  return true;
     	} 
        return false;
    }  

//--------------------

};

} // namespace ImgAlgos

#endif // IMGALGOS_IMGINTFORBINS_H
