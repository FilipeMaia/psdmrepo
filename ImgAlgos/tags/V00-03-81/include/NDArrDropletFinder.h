#ifndef IMGALGOS_NDARRDROPLETFINDER_H
#define IMGALGOS_NDARRDROPLETFINDER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class NDArrDropletFinder.
//
//------------------------------------------------------------------------

//-----------------
// Class Headers --
//-----------------

#include "psana/Module.h"
#include "MsgLogger/MsgLogger.h"

//#include "psalg/psalg.h"
#include "ImgAlgos/GlobalMethods.h"
#include "ImgAlgos/TimeInterval.h"
#include "ImgAlgos/AlgDroplet.h"
#include "ImgAlgos/AlgSmearing.h"

#include <cstddef>  // for size_t
#include <string>
#include <vector>

//#include <boost/shared_ptr.hpp>

//------------------------------------

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *  @ingroup ImgAlgos
 *
 *  @brief NDArrDropletFinder - psana module, finds "droplets" (wide peaks) in data ndarray and saves their list in output ndarray
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */

  //typedef boost::shared_ptr<AlgDroplet>  shpDF;
  //typedef boost::shared_ptr<AlgSmearing> shpSM;


  static const size_t N_WINDOWS_BLK = 50;

  struct WINDOW{
    size_t seg;
    size_t rowmin;
    size_t rowmax;
    size_t colmin;
    size_t colmax;
  };


class NDArrDropletFinder : public Module {
public:

  typedef float droplet_t;

  // Default constructor
  NDArrDropletFinder (const std::string& name) ;

  // Destructor
  virtual ~NDArrDropletFinder () ;

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

protected:

private:

  void   printInputPars();
  string getCommonFileName(Event& evt);
  void   parse_windows_pars();
  void   print_windows();

  void   printWarningMsg(const std::string& add_msg=std::string());
  void   initProc(Event& evt, Env& env);
  void   procEvent(Event& evt, Env& env);
  void   printFoundNdarray();

  void   appendVectorOfDroplets(const std::vector<AlgDroplet::Droplet>& v);
  void   saveDropletsInEvent(Event& evt);
  void   saveDropletsInFile(Event& evt);
  //  void   saveNDArrInFile(Event& evt);

  Pds::Src    m_src;
  Source      m_source;
  std::string m_key;
  std::string m_key_out;
  std::string m_key_sme;
  double      m_thr_low;
  double      m_thr_high;
  double      m_sigma;       // smearing sigma in pixel size
  int         m_nsm;         // number of pixels for smearing [i0-m_nsm, i0+m_nsm]
  int         m_rpeak;       // number of pixels for peak finding in [i0-m_npeak, i0+m_npeak]
  double      m_low_value;   // low value of intensity of pixels below threshold and outside windows
  std::string m_windows;     // windows for processing
  std::string m_ofname_pref; // output file name common prefix
  unsigned    m_print_bits;
  unsigned    m_count_evt;
  unsigned    m_count_get;
  unsigned    m_count_msg;
  unsigned    m_count_sel;

  std::string m_str_exp;

  unsigned        m_ndim;             // ndarray number of dimensions
  DATA_TYPE       m_dtype;            // numerated data type for data array
  DETECTOR_TYPE   m_dettype;          // numerated detector type source
  bool            m_isconst;

  size_t          m_nsegs;
  size_t          m_nrows;
  size_t          m_ncols;
  size_t          m_stride;

  TimeInterval   *m_time;

  std::vector<WINDOW>       v_windows;  // vector of the WINDOW structure with input parameters
  std::vector<AlgDroplet*>  v_algdf;    // vector of pointers to the AlgDroplet objects for windows
  std::vector<AlgSmearing*> v_algsm;    // vector of pointers to the AlgSmearing objects for windows
  std::vector<AlgDroplet::Droplet> v_droplets; // vector of droplets for entire event

//-------------------
//-------------------
//-------------------
//-------------------

  template <typename T, unsigned NDim>
    void initProcForNDArr( ndarray<const T,NDim>& nda )
    {
        m_count_get ++;

	// Retreive ndarray parameters
	//-----------------------------
        m_ndim    = NDim;
        m_dtype   = dataType<T>();
        if (m_print_bits & 1) printFoundNdarray();
	
	// std::cout << nda << '\n';

	const unsigned* shape = nda.shape();
	const int* strides = nda.strides();

        m_ncols  = shape[m_ndim-1];
        m_nrows  = shape[m_ndim-2];
	m_nsegs  = (m_ndim>2) ? nda.size()/m_ncols/m_nrows : 1;
	m_stride = (m_ndim>2) ? strides[m_ndim-3] : 1;

	if (m_print_bits & 256) {
            std::stringstream ss; ss << "Input parameters:";
	    ss << "size   :" << nda.size() << '\n'; 
	    ss << "shape  :"; for(unsigned i=0; i<m_ndim; ++i) ss << " " << shape[i];   ss << '\n';
	    ss << "strides:"; for(unsigned i=0; i<m_ndim; ++i) ss << " " << strides[i]; ss << '\n';

    	    ss << " m_ncols:"  << m_ncols
               << " m_nrows:"  << m_nrows 
               << " m_nsegs:"  << m_nsegs 
               << " m_stride:" << m_stride;

            MsgLog(name(), info, ss.str());  
	}

	unsigned pbits_sm = (m_print_bits &  64) ?     017 : 0;
	         pbits_sm = (m_print_bits & 128) ? 0177777 : pbits_sm;
	unsigned pbits_df = (m_print_bits &  64) ?     017 : 0;
	         pbits_df = (m_print_bits & 128) ? 0177777 : pbits_df;


	// Fill-in vectors of processing algorithms
	//------------------------------------------

        if (v_windows.empty()) {
            // All segments will be processed

            v_algdf.reserve(m_nsegs);
            v_algsm.reserve(m_nsegs);
	    
	    for(size_t seg=0; seg<m_nsegs; ++seg) {
	    
                AlgDroplet* p_df = new AlgDroplet ( m_rpeak, m_thr_low, m_thr_high, pbits_df,
	                    		            seg, 0, m_nrows, 0, m_ncols );
                v_algdf.push_back(p_df);
	        
	        
                AlgSmearing* p_sm = (m_sigma)? new AlgSmearing( m_sigma, m_nsm, m_thr_low, 0, pbits_sm,
								seg, 0, m_nrows, 0, m_ncols ) : 0;
                v_algsm.push_back(p_sm);
	        
	    }	    
            return;
	}


	// Windows will be processed
        v_algdf.reserve(v_windows.size());
        v_algsm.reserve(v_windows.size());

	// Check consistency of requested windows
        std::vector<WINDOW>::iterator it  = v_windows.begin();
        for ( ; it != v_windows.end(); ++it) {
	    if(it->seg >= m_nsegs) { 
              MsgLog(name(), warning, "Window segment number: " << it->seg 
                                      << " exceeds the (index)number of segments in ndarray: " << m_nsegs
	          			    << "\n    WINDOW IS IGNORED! FIX IT in *.cfg file");
	      continue;
	    }
	    
	    if(it->rowmin > m_nrows
	    || it->rowmax > m_nrows) {
              MsgLog(name(), warning, "Window number of rows min: " << it->rowmin << " or max: "  << it->rowmax 
                                      << " exceeds the number of rows in ndarray: " << m_nrows
	          			    << "\n    FIX IT in *.cfg file");
	    }
	    
	    if(it->colmin > m_ncols
	    || it->colmax > m_ncols) {
               MsgLog(name(), warning, "Window number of columns min: " << it->colmin << " or max: "  << it->colmax  
                                      << " exceeds the number of columns in ndarray: " << m_ncols
	          			    << "\n    FIX IT in *.cfg file");
	    }
	    
	    
	    AlgDroplet*  p_df = new AlgDroplet (m_rpeak, m_thr_low, m_thr_high, pbits_df,
	          				       it->seg, it->rowmin, it->rowmax, it->colmin, it->colmax );
            v_algdf.push_back(p_df);
	    
	    
            AlgSmearing* p_sm = (m_sigma) ? new AlgSmearing( m_sigma, m_nsm, m_thr_low, 0, pbits_sm,
							     it->seg, it->rowmin, it->rowmax, it->colmin, it->colmax, 
                                                             m_low_value ) : 0;
                                              
            v_algsm.push_back(p_sm);
	    
	} // for ( ;
    }

//-------------------

  template <typename T, unsigned NDim>
    void procEventForNDArr( Event& evt, ndarray<const T,NDim>& nda )
    {
        m_count_get ++;

        v_droplets.clear();
        v_droplets.reserve(AlgDroplet::NDROPLETSBLOCK);

        //m_ndim
        //m_dtype
	bool has_data = false;
	const T* data = nda.data();


	// create and pre-fill output ndarray for smearing
        ndarray<T,NDim> data_sme(nda.shape());
        T* p_data_sme = data_sme.data();

	if (! m_key_sme.empty() && m_sigma)
            std::fill_n(p_data_sme, int(nda.size()), T(m_low_value));
	    // if     (m_opt==0) std::fill_n(data_sme.data(), int(data.size()), T(0));
            // else if(m_opt==1) std::memcpy(data_sme.data(), data, data.size()*sizeof(T));


	if (m_print_bits & 128) {
          std::stringstream ss; ss << "m_count_get=" << m_count_get << " data:";
	  for(unsigned i=0; i<10; ++i) ss << " " << data[i];  ss << '\n'; 
          MsgLog(name(), info, ss.str());  
	}

        std::vector<AlgSmearing*>::iterator ism = v_algsm.begin();
        std::vector<AlgDroplet*>::iterator  idf = v_algdf.begin();
        for ( ; idf != v_algdf.end(); ++idf, ++ism) {

            //AlgDroplet*  p_df = (*idf);
            //AlgSmearing* p_sm = (*ism);
	    
	    size_t seg = (*idf)->segind();
	    
	    //cout << "df seg:" << seg << '\n';
	    
	    unsigned int shape[2] = {m_nrows, m_ncols};
            ndarray<const T,2> nda_raw(&data[seg*m_stride], shape);
            //ndarray<const T,2> nda_raw = make_ndarray(&data[seg*m_stride], m_nrows, m_ncols);
	    
	    if (m_sigma) {
                // Apply smearing before droplet-finder
	        //ndarray<T,2> nda_sme = make_ndarray<T>(m_nrows, m_ncols);
	        ndarray<T,2> nda_sme(shape);
	        (*ism)->smearing<T>(nda_raw, nda_sme);
	        has_data = (*idf)->findDroplets<T>(nda_sme);

		// Copy in order to save smeared ndarray in the file and event store
	        if (! m_key_sme.empty()) {
		  //std::memcpy(&p_data_sme[seg*m_stride], nda_sme.data(), nda_sme.size()*sizeof(T));

	          T* p_tot = &p_data_sme[seg*m_stride];
	          const T* p_seg = nda_sme.data();
		  for(unsigned i=0; i<nda_sme.size(); i++)
		    if (p_seg[i] != T(m_low_value)) p_tot[i] = p_seg[i];
		}
	    }
	    else {
	      has_data = (*idf)->findDroplets<T>(nda_raw);
	    }
	    
	    if (has_data) appendVectorOfDroplets( (*idf)->getDroplets() );
	}

	// Save here whatever need to be saved after processing

	//if (m_print_bits & 8) MsgLog(name(), info, "Total number of droplets found: " << v_droplets.size());
        if (v_droplets.size()) {
          saveDropletsInEvent(evt);
	  if(! m_ofname_pref.empty()) saveDropletsInFile(evt);
	}

	const std::vector<std::string>& comments = std::vector<std::string>();
        const unsigned pbits = (m_print_bits & 16) ? 0377 : 0;


	if(! m_ofname_pref.empty()) {
	  std::string fname = getCommonFileName(evt) + "-raw.txt";
	  saveNDArrayInFile<T,NDim>(nda, fname, comments, pbits);
	}


        if (! m_key_sme.empty() && m_sigma) {
          saveNDArrayInEvent<T,NDim>(evt, m_src, m_key_sme, data_sme);
	  if(! m_ofname_pref.empty()) {
            std::string fname = getCommonFileName(evt) + "-smeared.txt";
	    saveNDArrayInFile<T,NDim>(data_sme, fname, comments, pbits);
	  }
	}
    }

//-------------------
//-------------------
//-------------------
//-------------------

  template <typename T, unsigned NDim>
    bool initProcForTypeNDim(Event& evt)
    {
        m_ndim  = 0;
        m_dtype = NONDEFDT;

        // CONST
        shared_ptr< ndarray<const T,NDim> > shp_const = evt.get(m_source, m_key, &m_src);
        if (shp_const.get()) {
          m_isconst = true;
          initProcForNDArr<T,NDim>( *shp_const.get() );
 	  return true;
        } 

	// NON-CONST
        shared_ptr< ndarray<T,NDim> > shp = evt.get(m_source, m_key, &m_src);
        if (shp.get()) {
          m_isconst = false;
	  //ndarray<T,NDim>* pnda = shp.get(); 
	  //ndarray<const T,NDim> nda_const(pnda->data(), pnda->shape());
	  ndarray<const T,NDim> nda_const(shp->data(), shp->shape());
          initProcForNDArr<T,NDim>( nda_const );
 	  return true;
        } 

        return false;
    }

//-------------------

  template <typename T, unsigned NDim>
    bool procEventForTypeNDim(Event& evt)
    {
      if (m_isconst ) { // CONST
        shared_ptr< ndarray<const T,NDim> > shp_const = evt.get(m_source, m_key, &m_src);
        if (shp_const.get()) {
          procEventForNDArr<T,NDim>(evt, *shp_const.get());
 	  return true;
        } 
      }
      else { // NON-CONST
        shared_ptr< ndarray<T,NDim> > shp = evt.get(m_source, m_key, &m_src);
        if (shp.get()) {
	  ndarray<const T,NDim> nda_const(shp->data(), shp->shape());
          procEventForNDArr<T,NDim>(evt, nda_const);
 	  return true;
        } 
      }
      return false;
    }

//-------------------
//-------------------
//-------------------

  template <typename T>
    bool initProcForType(Event& evt)
    {
      if      ( initProcForTypeNDim<T,2> (evt) ) return true;
      else if ( initProcForTypeNDim<T,3> (evt) ) return true;
      else if ( initProcForTypeNDim<T,4> (evt) ) return true;
      else if ( initProcForTypeNDim<T,5> (evt) ) return true;
      return false;
    }

//-------------------

  template <typename T>
    bool procEventForType(Event& evt)
    {
      if      ( m_ndim==2 && procEventForTypeNDim<T,2> (evt) ) return true;
      else if ( m_ndim==3 && procEventForTypeNDim<T,3> (evt) ) return true;
      else if ( m_ndim==4 && procEventForTypeNDim<T,4> (evt) ) return true;
      else if ( m_ndim==5 && procEventForTypeNDim<T,5> (evt) ) return true;
      return false;
    }
 
//-------------------
//-------------------
//-------------------
};

} // namespace ImgAlgos

#endif // IMGALGOS_NDARRDROPLETFINDER_H
