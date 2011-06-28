#ifndef CSPAD_MOD_CSPADPEDESTALS_H
#define CSPAD_MOD_CSPADPEDESTALS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadPedestals.
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
#include "psddl_psana/cspad.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace cspad_mod {

/**
 *  @brief Psana module which calculates pedestals from dark CsPad run.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version \$Id$
 *
 *  @author Andy Salnikov
 */

class CsPadPedestals : public Module {
public:

  enum { MaxQuads = Psana::CsPad::MaxQuadsPerSensor };
  enum { MaxSectors = Psana::CsPad::SectorsPerQuad };
  enum { NumColumns = Psana::CsPad::ColumnsPerASIC };
  enum { NumRows = Psana::CsPad::MaxRowsPerASIC*2 };
  
  // Default constructor
  CsPadPedestals (const std::string& name) ;

  // Destructor
  virtual ~CsPadPedestals () ;

  /// Method which is called at the beginning of the run
  virtual void beginRun(Event& evt, Env& env);
  
  /// Method which is called with event data, this is the only required 
  /// method, all other methods are optional
  virtual void event(Event& evt, Env& env);
  
  /// Method which is called once at the end of the job
  virtual void endJob(Event& evt, Env& env);

protected:

  /// collect statistics
  void collectStat(unsigned qNum, const uint16_t* data);
  
private:

  std::string m_pedFile;
  std::string m_noiseFile;
  
  Pds::Src m_src; // source address of the data object
  
  unsigned m_segMask[MaxQuads];  // segment masks per quadrant
  
  unsigned long m_count;  // number of events seen
  double m_sum[MaxQuads][MaxSectors][NumColumns][NumRows];   // sum per pixel
  double m_sum2[MaxQuads][MaxSectors][NumColumns][NumRows];  // sum of squares per pixel
  
};

} // namespace cspad_mod

#endif // CSPAD_MOD_CSPADPEDESTALS_H
