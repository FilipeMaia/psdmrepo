//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ExpNameDatabase...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ExpNameDb/ExpNameDatabase.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream>
#include <boost/format.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ExpNameDb/Exceptions.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char logger[] = "ExpNameDatabase";
}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ExpNameDb {

//----------------
// Constructors --
//----------------
ExpNameDatabase::ExpNameDatabase (const std::string fname)
  : m_path(fname)
{
  if (m_path.path().empty()) {
    MsgLog(logger, error, "Failed to find database file " << fname);
    throw FileNotFoundError(ERR_LOC, fname);
  }
}

/**
 *  @brief Get instrument and experiment name given experiment ID.
 *
 *  @param[in] id    Experiment ID.
 *  @return Pair of strings, first string is instrument name, second is experiment name,
 *         both will be empty if ID is not known.
 *
 */
std::pair<std::string, std::string>
ExpNameDatabase::getNames(unsigned id) const
{
  // open file and read it
  std::ifstream db(m_path.path().c_str());
  unsigned dbExpNum;
  std::string instrName;
  std::string expName;
  std::pair<std::string, std::string> res;
  while (db >> dbExpNum >> instrName >> expName) {
    if (dbExpNum == id) {
      res = std::make_pair(instrName, expName);
      break;
    }
  }
  MsgLog(logger, debug, boost::format("ExpNameDatabase::getNames(%1%) -> (%2%, %3%)") % id % res.first % res.second);
  return res;
}

/**
 *  @brief Get experiment ID given instrument and experiment names.
 *
 *  Instrument name may be empty, if experiment name is unambiguous. If instrument name
 *  is empty and experiment name is ambiguous then first matching ID is returned.
 *
 *  @param[in] instrument   Instrument name.
 *  @param[in] experiment   Experiment name.
 *  @return Experiment ID or 0 if instrument/experiment is not known.
 *
 */
unsigned
ExpNameDatabase::getID(const std::string& instrument, const std::string& experiment) const
{
  // open file and read it
  std::ifstream db(m_path.path().c_str());
  unsigned dbExpNum;
  std::string instrName;
  std::string expName;
  unsigned res = 0;
  while (db >> dbExpNum >> instrName >> expName) {
    if (expName == experiment and (instrument.empty() or instrName == instrument)) {
      res = dbExpNum;
      break;
    }
  }
  MsgLog(logger, debug, boost::format("ExpNameDatabase::getID(%1%, %2%) -> %3%") % instrument % experiment % res);
  return res;
}

/**
 *  @brief Get instrument name and experiment ID for given experiment name.
 *
 *  If experiment name is ambiguous then first matching name and ID is returned.
 *
 *  @param[in] experiment   Experiment name.
 *  @return Pair of instrument name and experiment ID, name will be empty if experiment is not known.
 *
 */
std::pair<std::string, unsigned>
ExpNameDatabase::getInstrumentAndID(const std::string& experiment) const
{
  // open file and read it
  std::ifstream db(m_path.path().c_str());
  unsigned dbExpNum;
  std::string instrName;
  std::string expName;
  std::pair<std::string, unsigned> res;
  while (db >> dbExpNum >> instrName >> expName) {
    if (expName == experiment) {
      res = std::make_pair(instrName, dbExpNum);
    }
  }
  MsgLog(logger, debug, boost::format("ExpNameDatabase::getInstrumentAndID(%1%) -> (%2%, %3%)") % experiment % res.first % res.second);
  return res;
}


} // namespace ExpNameDb
