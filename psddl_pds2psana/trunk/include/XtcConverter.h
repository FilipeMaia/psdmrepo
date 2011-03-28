#ifndef PSDDL_PDS2PSANA_XTCCONVERTER_H
#define PSDDL_PDS2PSANA_XTCCONVERTER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcConverter.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Xtc.hh"
#include "PSEvt/Event.h"
#include "PSEnv/ConfigStore.h"
#include "PSEnv/EpicsStore.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace psddl_pds2psana {

/**
 *  @brief Class responsible for conversion of XTC objects into 
 *  real psana objects.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class XtcConverter  {
public:

  // Default constructor
  XtcConverter () ;

  // Destructor
  ~XtcConverter () ;
  
  /**
   *  @brief Convert one object and store it in the event.
   */
  void convert(const boost::shared_ptr<Pds::Xtc>& xtc, PSEvt::Event& evt, PSEnv::ConfigStore& cfgStore);
  
  /**
   *  @brief Convert one object and store it in the config store.
   */
  void convertConfig(const boost::shared_ptr<Pds::Xtc>& xtc, PSEnv::ConfigStore& cfgStore);

  /**
   *  @brief Convert one object and store it in the epics store.
   */
  void convertEpics(const boost::shared_ptr<Pds::Xtc>& xtc, PSEnv::EpicsStore& eStore);

protected:

private:

  // Data members

};

} // namespace psddl_pds2psana

#endif // PSDDL_PDS2PSANA_XTCCONVERTER_H
