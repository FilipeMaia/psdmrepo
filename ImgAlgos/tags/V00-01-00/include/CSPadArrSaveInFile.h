#ifndef IMGALGOS_CSPADARRSAVEINFILE_H
#define IMGALGOS_CSPADARRSAVEINFILE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadArrSaveInFile.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "ImgAlgos/CSPadBaseModule.h"

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

namespace ImgAlgos {

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

class CSPadArrSaveInFile : public CSPadBaseModule {
public:

    enum { MaxQuads   = Psana::CsPad::MaxQuadsPerSensor }; // 4
    enum { MaxSectors = Psana::CsPad::SectorsPerQuad    }; // 8
    enum { NumColumns = Psana::CsPad::ColumnsPerASIC    }; // 185 THERE IS A MESS IN ONLINE COLS<->ROWS
    enum { NumRows    = Psana::CsPad::MaxRowsPerASIC*2  }; // 388 THERE IS A MESS IN ONLINE COLS<->ROWS 
    enum { SectorSize = NumColumns * NumRows            }; // 185 * 388
  
  // Default constructor
  CSPadArrSaveInFile (const std::string& name) ;

  // Destructor
  virtual ~CSPadArrSaveInFile () ;

  /// Method which is called once at the beginning of the job
  virtual void beginJob(Event& evt, Env& env);
  
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

  void procQuad(unsigned quad, const int16_t* data);
  void printInputParameters();
  void printEventId(Event& evt);
  void printTimeStamp(Event& evt);
  std::string strEventCounter();
  std::string strTimeStamp(Event& evt);
  std::string strRunNumber(Event& evt);
  std::string strTimeDependentFileName(Event& evt);
  void saveInFile(Event& evt); 
  template <typename T>
  void saveCSPadArrayInFile(std::string& fname, T arr[MaxQuads][MaxSectors][NumColumns][NumRows]);


private:
  std::string    m_outFile;
  unsigned       m_print_bits;   
  unsigned long  m_count;  // number of events from the beginning of job

  int16_t        m_arr    [MaxQuads][MaxSectors][NumColumns][NumRows];  // array for all cspad pixels
};

} // namespace ImgAlgos

#endif // IMGALGOS_CSPADARRSAVEINFILE_H
