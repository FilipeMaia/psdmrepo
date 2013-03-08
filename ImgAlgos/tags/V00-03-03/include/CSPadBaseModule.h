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

protected:

  /// Returns the source address of cspad device which was found
  const Pds::Src& source() const { return m_src; }

  /// Returns the source address of cspad device as specified in configuration (or constructor)
  const Source& sourceConfigured() const { return m_str_src; }

  /// Returns the source address of cspad device
  const std::string& inputKey() const { return m_key; }

  /// Returns the source address of cspad device
  unsigned segMask(int seg) const { return m_segMask[seg]; }

private:

  Source         m_str_src;         // string with source name
  std::string    m_key;             // string with key name
  Pds::Src       m_src;             // source address of the data object
  unsigned       m_segMask[Psana::CsPad::MaxQuadsPerSensor];  // segment masks per quadrant

};

} // namespace ImgAlgos

#endif // IMGALGOS_CSPADBASEMODULE_H
