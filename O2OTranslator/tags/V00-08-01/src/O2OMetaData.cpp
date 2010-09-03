//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OMetaData...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/O2OMetaData.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
O2OMetaData::O2OMetaData ( unsigned long runNumber,
                           const std::string& runType,
                           const std::string& instrument,
                           const std::string& experiment,
                           const std::list<std::string>& extraMetaData )
  : m_runNumber(runNumber)
  , m_runType(runType)
  , m_instrument(instrument)
  , m_experiment(experiment)
  , m_extraMetaData()
{
  typedef std::list<std::string>::const_iterator MDIter ;
  for ( MDIter it = extraMetaData.begin() ; it != extraMetaData.end() ; ++ it ) {
    const std::string& nameValue = *it ;
    std::string::size_type c = nameValue.find(':') ;
    std::string name = c == std::string::npos ? nameValue : std::string(nameValue,0,c) ;
    std::string value = c == std::string::npos ? std::string() : std::string(nameValue,c+1) ;
    m_extraMetaData.insert( cont_type::value_type(name,value) ) ;
  }
}

//--------------
// Destructor --
//--------------
O2OMetaData::~O2OMetaData ()
{
}

} // namespace O2OTranslator
