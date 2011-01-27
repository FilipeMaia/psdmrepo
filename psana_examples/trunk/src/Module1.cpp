//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Module1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/Module1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(Module1)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

//----------------
// Constructors --
//----------------
Module1::Module1 (const std::string& name)
  : Module(name)
  , m_count(0)
  , m_maxEvents()
  , m_filter()
{
  // get the value from configuration, use default
  m_maxEvents = config("events", 32U);
  m_filter = config("filter", false);
}

//--------------
// Destructor --
//--------------
Module1::~Module1 ()
{
}

/// Method which is called with event data
void 
Module1::event(Event& evt, Env& env)
{
  ++m_count;
  MsgLogRoot(info, name() << ": processing event #" << m_count);
  
  if (m_filter && m_count % 10 == 0) skip();
  if (m_count >= m_maxEvents) stop();
}

