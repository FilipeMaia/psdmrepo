#ifndef IMGALGOS_CSPADBASEMODULE_H
#define IMGALGOS_CSPADBASEMODULE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadBaseModule.
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

namespace ImgAlgos {

/// @addtogroup ImgAlgos

/**
 *  @ingroup ImgAlgos
 *
 *  @brief Base class for many psana modules working with cspad.
 *
 *  The purpose of this base class is to do some common work which is done
 *  by almost every module which works with cspad data. In particular
 *  it implements beginRun() method which finds cspad configuration objects,
 *  saves exact address of the cspad device, and also fills  segment mask
 *  array.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Andy Salnikov
 */

class CSPadBaseModule : public Module {
public:

  const static int MaxQuads   = Psana::CsPad::MaxQuadsPerSensor; // 4
  const static int MaxSectors = Psana::CsPad::SectorsPerQuad;    // 8
  const static int NumColumns = Psana::CsPad::ColumnsPerASIC;    // 185 THERE IS A MESS IN ONLINE COLS<->ROWS
  const static int NumRows    = Psana::CsPad::MaxRowsPerASIC*2;  // 388 THERE IS A MESS IN ONLINE COLS<->ROWS 
  const static int SectorSize = NumColumns * NumRows;            // 185 * 388

  // Default constructor
  CSPadBaseModule(const std::string& name,
      const std::string& keyName="key",
      const std::string& defKey="",
      const std::string& sourceName="source",
      const std::string& defSource="DetInfo(:Cspad)");

  // Destructor
  virtual ~CSPadBaseModule () ;

  /**
   *   @brief Method which is called at the beginning of the run
   *
   *   This implementation finds all cspad configuration objects present in
   *   the environment using the source address and key provided in constructor
   *   or in configuration. If 0 or more than one objects are found it will
   *   return with "terminate" flag and psana will stop. Otherwise it will
   *   remember actual device address (availabale later from source() method)
   *   and will also fill segment masks from configuration objects (use
   *   segMask(i) to get it).
   */
  virtual void beginRun(Event& evt, Env& env);

  /// Initialization before data processing in event()
  virtual void initData();

  /// Quad data processing in event()
  virtual void procQuad(unsigned quad, const int16_t* data);

  /// Summary of the data processing in event()
  virtual void summaryData(Event& evt);

  /// Prints values of the base module parameters
  void printBaseParameters();

protected:

  /// Returns the source address of cspad device which was found
  const Pds::Src& source() const { return m_src; }

  /// Returns the source address of cspad device as specified in configuration (or constructor)
  const Source& sourceConfigured() const { return m_str_src; }

  /// Returns the source address of cspad device
  const std::string& inputKey() const { return m_key; }

  /// Returns the source address of cspad device
  unsigned segMask(int seg) const { return m_segMask[seg]; }

  /// Returns the counter of found data records
  unsigned counter() const { return m_count; }

private:

  Source         m_str_src;         // string with source name
  std::string    m_key;             // string with key name
  Pds::Src       m_src;             // source address of the data object
  unsigned       m_segMask[Psana::CsPad::MaxQuadsPerSensor];  // segment masks per quadrant
  unsigned       m_count;  

public:

  /**
   * @brief Loop over quads, get data for types TDATA and TELEMENT and call procQuad(...).
   * 
   */

  template <typename TDATA, typename TELEMENT>
  bool procEventForType(Event& evt) {

      shared_ptr<TDATA> shp_data = evt.get(source(), inputKey());
      if (shp_data.get()) {
      
        ++ m_count;
        initData();                                             // <<===========
        
        int nQuads = shp_data->quads_shape()[0];
        for (int iq = 0; iq != nQuads; ++ iq) {
          
          const TELEMENT& quad = shp_data->quads(iq);
          const ndarray<const int16_t, 3>& data = quad.data();
          procQuad(quad.quad(), data.data());                   // <<===========
        } 

        summaryData(evt);                                       // <<===========

        return true;
      }
      return false;
  }

};

} // namespace ImgAlgos

#endif // IMGALGOS_CSPADBASEMODULE_H
