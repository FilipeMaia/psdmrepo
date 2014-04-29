#ifndef IMGALGOS_IMGSPECTRA_H
#define IMGALGOS_IMGSPECTRA_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgSpectra.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
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
 *  @brief ImgSpectra extracts two spectra from image and evaluate their relative difference.
 *
 *  ImgSpectra psana module:
 *  - gets the image object from event,
 *  - selects two spectral band regions and integrates amplitudes for each column,
 *  - saves two spectra, f_sig and f_ref and their relative difference 
 *    as a ndarray<double,2> object in the event. 
 * 
 *  This software was developed for the LCLS project. If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see CSPadImageProducer
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class ImgSpectra : public Module {
public:

  // Default constructor
  ImgSpectra (const std::string& name) ;

  // Destructor
  virtual ~ImgSpectra () ;

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
  void printEventRecord(Event& evt, std::string=std::string());
  void procEvent(Event& evt);
  //void getSpectrum(float rowc, float tilt, unsigned width, unsigned ind);
  void difSpectrum();
  void printSpectra(Event& evt);

private:

  Pds::Src    m_src;
  Source      m_str_src;     // i.e. Opal:
  std::string m_key_in;      // input key
  std::string m_key_out;     // output key
  float       m_sig_rowc;    // signal    band row center coordinate
  float       m_ref_rowc;    // reference ...
  float       m_sig_tilt;    // signal    band tilt angle
  float       m_ref_tilt;    // reference ...
  unsigned    m_sig_width;   // signal    band width in number of rows
  unsigned    m_ref_width;   // signal    ...
  unsigned    m_print_bits;
  long        m_count;

  unsigned    m_cols;
  ndarray<double, 2> m_data;

protected:

//--------------------

    template <typename T>
    void getSpectrum(const T* img_data, float rowc, float tilt, unsigned width, unsigned ind)
    {
      for( unsigned c=0; c<m_cols; c++ ) {
	int row_min = int(rowc - tilt * c - 0.5*width);
	int row_max = row_min + width + 1;

	for( int r=row_min; r<row_max; r++ ) {

	  m_data[ind][c] += img_data[r*m_cols + c];
	}
      }
    }

//--------------------

    template <typename T>
    void retrieveSpectra(const ndarray<const T,2>& sp_ndarr, bool print_msg=false)
      {
        const T* img_data = sp_ndarr.data();               // Access to entire image
	unsigned rows = sp_ndarr.shape()[0];
	m_cols = sp_ndarr.shape()[1];
	if(print_msg) MsgLog( name(), info, "Image shape =" << rows << ", " << m_cols);

	if(m_data.empty()) {
          m_data = make_ndarray<double>(3, m_cols); // memory allocation for output array of spectra
	}

        std::fill_n(m_data.begin(), m_data.size(), double(0));

        getSpectrum(img_data, m_sig_rowc, m_sig_tilt, m_sig_width, 0);
        getSpectrum(img_data, m_ref_rowc, m_ref_tilt, m_ref_width, 1);	
        difSpectrum();	
      }

//--------------------
//--------------------
//--------------------
};

} // namespace ImgAlgos

#endif // IMGALGOS_IMGSPECTRA_H
