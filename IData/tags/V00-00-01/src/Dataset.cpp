//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Dataset...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "IData/Dataset.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ExpNameDb/ExpNameDatabase.h"
#include "IData/Exceptions.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "Dataset";

  // parse experiment name
  void parseExpName(const std::string& exp, unsigned& expId, std::string& instrName, std::string& expName);

  // parse run list
  void parseRuns(const std::string& str, IData::Dataset::Runs& runs);

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace IData {

// static data members
Dataset::Key2Val Dataset::s_key2val;      ///< Application-wide options
unsigned Dataset::s_expId(0);             ///< Application-wide experiment ID
std::string Dataset::s_instrName;         ///< Application-wide instrument name
std::string Dataset::s_expName;           ///< Application-wide experiment name


/**
 *  @brief Sets application-wide experiment name.
 *
 *  Experiment name can be specified with the syntax acceptable for exp key.
 *  Individual datasets can override application-wide value.
 *
 *  @param[in] exp  new application-wide experiment name
 *
 *  @throw ExpNameException thrown if specified name is not found
 */
void
Dataset::setAppExpName(const std::string& exp)
{
  ::parseExpName(exp, s_expId, s_instrName, s_expName);
  s_key2val["exp"] = exp;
}

/**
 *  @brief Sets default application-wide option.
 *
 *  Sets default application-wide value for an option.
 *  Individual datasets can override application-wide values.
 *
 *  @param[in] key   Key name
 *  @param[in] value New application-wide value for this key
 */
void
Dataset::setDefOption(const std::string& key, const std::string& value)
{
  if (key == "exp") {
    ::parseExpName(value, s_expId, s_instrName, s_expName);
  } else if (key == "run") {
    MsgLog(logger, warning, "setDefOption() does not accept run numbers");
  }
  s_key2val[key] = value;
}

//----------------
// Constructors --
//----------------
Dataset::Dataset()
  : m_key2val()
  , m_runs()
  , m_expId(0)
  , m_instrName()
  , m_expName()
{
}

Dataset::Dataset(const std::string& ds)
  : m_key2val()
  , m_runs()
  , m_expId(0)
  , m_instrName()
  , m_expName()
{
  // split it at colons
  std::vector<std::string> options;
  boost::split(options, ds, boost::is_any_of(":"), boost::token_compress_on);
  for (std::vector<std::string>::const_iterator it = options.begin(); it != options.end(); ++ it) {

    std::string option = *it;
    boost::trim(option);
    if (option.empty()) continue;

    std::string key(option);
    std::string val;

    std::string::size_type p = option.find('=');
    if (p != std::string::npos) {
      key.erase(p);
      boost::trim(key);
      val = option.substr(p+1);
      boost::trim(val);
    }

    if (key == "exp") {
      ::parseExpName(val, m_expId, m_instrName, m_expName);
    } else if (key == "run") {
      ::parseRuns(val, m_runs);
    }

    m_key2val[key] = val;

  }

}

//--------------
// Destructor --
//--------------
Dataset::~Dataset()
{
}

/**
 *  @brief Returns true if the key is defined.
 *
 *  Key may be defined by either constructor or with a default
 *  application-wide option.
 *
 *  @param[in] key  Key name
 */
bool
Dataset::exists(const std::string& key) const
{
  return m_key2val.find(key) != m_key2val.end() or
      s_key2val.find(key) != s_key2val.end();
}

/**
 *  @brief Returns value for given key or empty string.
 *
 *  @param[in] key  Key name
 */
const std::string&
Dataset::value(const std::string& key) const
{
  // check my keys first
  Key2Val::const_iterator it = m_key2val.find(key);
  if (it != m_key2val.end()) return it->second;

  // check default keys
  it = s_key2val.find(key);
  if (it != s_key2val.end()) return it->second;

  // otherwise return empty string
  static std::string def;
  return def;
}

/// Returns experiment ID or 0 if it has not been defined.
unsigned
Dataset::expID() const
{
  if (m_expId == 0) return s_expId;
  return m_expId;
}

/// Returns instrument name or empty string if it has not been defined.
const std::string&
Dataset::instrument() const
{
  if (m_instrName.empty()) return s_instrName;
  return m_instrName;
}

/// Returns experiment name or empty string if it has not been defined.
const std::string&
Dataset::experiment() const
{
  if (m_expName.empty()) return s_expName;
  return m_expName;
}

/// Returns set of run numbers
const Dataset::Runs&
Dataset::runs() const
{
  return m_runs;
}

} // namespace IData

namespace {

// parse experiment name
void parseExpName(const std::string& exp, unsigned& expId, std::string& instrName, std::string& expName)
{
  ExpNameDb::ExpNameDatabase namedb;

  if (boost::all(exp, boost::is_digit())) {

    // all digits, must be experiment id
    unsigned num = boost::lexical_cast<unsigned>(exp);
    std::pair<std::string, std::string> instrExp = namedb.getNames(num);
    if (instrExp.first.empty()) {
      throw IData::ExpNameException(ERR_LOC, exp);
    }

    expId = num;
    instrName = instrExp.first;
    expName = instrExp.second;

  } else {

    // experiment name optionally with instrument name
    std::string::size_type p = exp.find('/');
    if (p == std::string::npos) {

      // only experiment name is given
      std::pair<std::string, unsigned> instrExp = namedb.getInstrumentAndID(exp);
      if (instrExp.first.empty()) {
        throw IData::ExpNameException(ERR_LOC, exp);
      }

      expId = instrExp.second;
      instrName = instrExp.first;
      expName = exp;

    } else {

      // both instrument ans experiment name is given
      const std::string instrument(exp, 0, p);
      const std::string experiment(exp, p+1);

      unsigned num = namedb.getID(instrument, experiment);
      if (num == 0) {
        throw IData::ExpNameException(ERR_LOC, exp);
      }

      expId = num;
      instrName = instrument;
      expName = experiment;

    }


  }


}

// parse run list
void parseRuns(const std::string& str, IData::Dataset::Runs& runs)
{
  runs.clear();

  // split it at commas
  std::vector<std::string> ranges;
  boost::split(ranges, str, boost::is_any_of(","), boost::token_compress_on);
  for (std::vector<std::string>::const_iterator it = ranges.begin(); it != ranges.end(); ++ it) {

    std::string range = *it;
    boost::trim(range);
    if (range.empty()) continue;

    std::string startStr(range);
    std::string endStr;

    std::string::size_type p = range.find('-');
    if (p != std::string::npos) {
      startStr.erase(p);
      boost::trim(startStr);
      endStr = range.substr(p+1);
      boost::trim(endStr);
    }

    unsigned start, end;
    try {
      start = boost::lexical_cast<unsigned>(startStr);
      if (endStr.empty()) {
        end = start;
      } else {
        end = boost::lexical_cast<unsigned>(endStr);
      }
    } catch (const boost::bad_lexical_cast& ex) {
      throw IData::RunNumberSpecException(ERR_LOC, str, ex.what());
    }

    runs.push_back(IData::Dataset::Runs::value_type(start, end));

  }

}

}
