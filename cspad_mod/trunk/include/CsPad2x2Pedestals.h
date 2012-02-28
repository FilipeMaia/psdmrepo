#ifndef CSPAD_MOD_CSPAD2X2PEDESTALS_H
#define CSPAD_MOD_CSPAD2X2PEDESTALS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2Pedestals.
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
#include "psddl_psana/cspad2x2.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace cspad_mod {

/**
 *  @brief Psana module which calculates 2x2 pedestals from dark CsPad2x2 run.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Andy Salnikov
 */

class CsPad2x2Pedestals : public Module {
public:

  enum { MaxSectors = Psana::CsPad2x2::SectorsPerQuad };
  enum { NumColumns = Psana::CsPad2x2::ColumnsPerASIC };
  enum { NumRows = Psana::CsPad2x2::MaxRowsPerASIC*2 };
  
  // Default constructor
  CsPad2x2Pedestals (const std::string& name) ;

  // Destructor
  virtual ~CsPad2x2Pedestals () ;

  /// Method which is called at the beginning of the run
  virtual void beginRun(Event& evt, Env& env);
  
  /// Method which is called with event data, this is the only required 
  /// method, all other methods are optional
  virtual void event(Event& evt, Env& env);
  
  /// Method which is called once at the end of the job
  virtual void endJob(Event& evt, Env& env);

protected:

  /// collect statistics
  void collectStat(const int16_t* data);
  
private:

  std::string m_pedFile;
  std::string m_noiseFile;
  
  Pds::Src m_src; // source address of the data object
  
  unsigned long m_count;  // number of events seen
  double m_sum[NumColumns][NumRows][MaxSectors];   // sum per pixel
  double m_sum2[NumColumns][NumRows][MaxSectors];  // sum of squares per pixel
  
};

} // namespace cspad_mod

#endif // CSPAD_MOD_CSPAD2X2PEDESTALS_H
