#ifndef IMGALGOS_USDUSBENCODERFILTER_H
#define IMGALGOS_USDUSBENCODERFILTER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class UsdUsbEncoderFilter.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <stdint.h> // uint8_t, uint32_t, etc.
#include <iomanip>  // for setw, setfill

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Module.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

namespace ImgAlgos {

  class TimeCode {
  
  public:

    typedef uint32_t tstamp_t;
    typedef uint8_t  code_t;
    typedef unsigned evnum_t;

    TimeCode() 
      : m_tst_sec(0)
      , m_tst_nsec(0)
      , m_code(0)
      , m_evnum(0)
    { 
      //MsgLog("TimeCode", info, "Default c-tor");
    }

    TimeCode(const tstamp_t& tst_sec, const tstamp_t& tst_nsec, const code_t& code, const evnum_t& evnum) 
      : m_tst_sec(tst_sec)
      , m_tst_nsec(tst_nsec)
      , m_code(code)
      , m_evnum(evnum)
    { 
      //MsgLog("TimeCode", info, "Regular c-tor");
    }

    const tstamp_t tst_sec()  const { return m_tst_sec; }
    const tstamp_t tst_nsec() const { return m_tst_nsec; }
    const code_t   code()     const { return m_code; }
    const evnum_t  evnum()    const { return m_evnum; }

    void set_tst_sec (const tstamp_t& tst_sec)  { m_tst_sec = tst_sec; }
    void set_tst_nsec(const tstamp_t& tst_nsec) { m_tst_nsec = tst_nsec; }
    void set_code    (const code_t&   code)     { m_code   = code; }
    void set_evnum   (const evnum_t&  evnum)    { m_evnum  = evnum; }

    void reset_tcode() {
      m_tst_sec = 0;
      m_tst_nsec = 0;
      m_code = 0;
      m_evnum = 0;
    }

    void set_tcode(const tstamp_t& tst_sec, const tstamp_t& tst_nsec, const code_t& code, const evnum_t& evnum) {
      m_tst_sec  = tst_sec;
      m_tst_nsec = tst_nsec;
      m_code     = code;
      m_evnum    = evnum;
    }

    // comparison for sorting
    bool operator<(const TimeCode& other) const {
      if (m_tst_sec != other.m_tst_sec) return m_tst_sec < other.m_tst_sec;
      return m_code < other.m_code;
    }

    bool operator==(const TimeCode& other) const {
      if (m_tst_sec != other.m_tst_sec) return false;
      if (abs(m_tst_nsec-other.m_tst_sec) > 1e6) return false;
      return m_code == other.m_code;
    }

    //bool operator>(const TimeCode& other) const {
    //  if (m_tst_sec != other.m_tst_sec) return m_tst_sec > other.m_tst_sec;
    //  return m_code > other.m_code;
    //}

  private:
    tstamp_t  m_tst_sec;
    tstamp_t  m_tst_nsec;
    code_t    m_code;
    evnum_t   m_evnum;
  }; // class


  std::ostream& operator<<(std::ostream& out, const TimeCode& tc) {
    return out << tc.tst_sec() << std::right << std::setw(10) << tc.tst_nsec() << std::setw(4) << int(tc.code()) << std::right << std::setw(6) << tc.evnum();
  }


  std::istream& operator>>(std::istream& in, TimeCode& tc) {
    TimeCode::tstamp_t tst_sec;
    TimeCode::tstamp_t tst_nsec;
    unsigned ucode; // !!! For some reason i/o does not work for uint8_t !!! 
    TimeCode::evnum_t  evnum;
    in >> tst_sec >> tst_nsec >> ucode >> evnum;
    tc.set_tst_sec(tst_sec);
    tc.set_tst_nsec(tst_nsec);
    tc.set_code(TimeCode::code_t(ucode));
    tc.set_evnum(evnum);
    return in;
  }

//----------------

//		---------------------
// 		-- Class Interface --
//		---------------------


/// @addtogroup ImgAlgos

/**
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

class UsdUsbEncoderFilter : public Module {
public:

  typedef TimeCode::tstamp_t tstamp_t;
  typedef TimeCode::code_t   code_t;
  typedef TimeCode::evnum_t  evnum_t;

  // Default constructor
  UsdUsbEncoderFilter (const std::string& name) ;

  // Destructor
  virtual ~UsdUsbEncoderFilter () ;

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
  Source   m_source;
  int      m_mode;
  std::string m_ifname;
  std::string m_ofname;
  uint8_t  m_bitmask;
  unsigned m_print_bits;
  unsigned m_count_evt;
  unsigned m_selected;
  std::ofstream* m_out;

  //TimeCode m_tc; 
  std::vector<TimeCode> v_tcode;
  std::vector<TimeCode>::const_iterator v_tc_iter;

  void loadFile();
  void printTimeCodeVector();
  void printInputParameters();
  void printData(Event& evt);
  void printConfig(Env& env);
  bool eventIsSelected(Event& evt, Env& env);
  std::string str_current_time();

  //void parseEvtString();
  //void parseOneRecord(char* rec);
};

} // namespace ImgAlgos

#endif // IMGALGOS_USDUSBENCODERFILTER_H
