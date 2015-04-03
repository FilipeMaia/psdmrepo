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
#include <cctype>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/format.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ExpNameDb/ExpNameDatabase.h"
#include "IData/Exceptions.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace fs = boost::filesystem;

namespace {

  const char* logger = "IData.Dataset";

  // parse experiment name or id, returns true on success
  bool parseExpName(const std::string& exp, unsigned& expId, std::string& instrName, std::string& expName);

  // parse run list
  void parseRuns(const std::string& str, IData::Dataset::Runs& runs);

  // parse run list
  void parseStreams(const std::string& str, IData::Dataset::Streams& streams);
 
  // checks to see if the string is a file name
  bool isFileName(const std::string& str);

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
  if (not ::parseExpName(exp, s_expId, s_instrName, s_expName)) {
    throw IData::ExpNameException(ERR_LOC, exp);
  }
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
    if (not ::parseExpName(value, s_expId, s_instrName, s_expName)) {
      throw IData::ExpNameException(ERR_LOC, value);
    }
  } else if (key == "run") {
    MsgLog(logger, warning, "setDefOption() does not accept run numbers");
  }
  s_key2val[key] = value;
}

//----------------
// Constructors --
//----------------
Dataset::Dataset()
  : m_isFile(false)
  , m_key2val()
  , m_runs()
  , m_expId(0)
  , m_instrName()
  , m_expName()
  , m_files()
{
}

Dataset::Dataset(const std::string& ds)
  : m_isFile(false)
  , m_key2val()
  , m_runs()
  , m_expId(0)
  , m_instrName()
  , m_expName()
  , m_files()
{
  if (::isFileName(ds)) {
 
    // parse file names with good extensions
    if (boost::ends_with(ds, ".xtc")) {
      parseXtcFileName(ds);
    } else if (boost::ends_with(ds, ".h5")) {
      parseHdfFileName(ds);
    }
    
    // store the file name
    m_files.push_back(ds);
 
    m_isFile = true;
    
  } else {

    // must be a dataset, split it at colons
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
        if (not ::parseExpName(val, m_expId, m_instrName, m_expName)) {
          throw IData::ExpNameException(ERR_LOC, val);
        }
      } else if (key == "run") {
        ::parseRuns(val, m_runs);
      } else if (key == "stream") {
        ::parseStreams(val, m_streams);
      }
      m_key2val[key] = val;

    }
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

/// Returns set of stream ranges
const Dataset::Streams&
Dataset::streams() const
{
  return m_streams;
}

/// Return the directory name for files
std::string 
Dataset::dirName() const
{
  // get directory name where to look for files
  std::string dir = this->value("dir");
  if (dir.empty()) {
    const char* type = this->exists("h5") ? "hdf5" : "xtc";
    boost::format fmt("%1%/%2%/%3%/%4%");
    const char* datadir = getenv("SIT_PSDM_DATA");
    if (datadir) {
      fmt % datadir % instrument() % experiment() % type;
      dir = fmt.str();
    } else {
      fmt % "/reg/d/psdm" % instrument() % experiment() % type;
      dir = fmt.str();
    }
  }
  return dir;
}

/// Return the list of file names for this dataset
const Dataset::NameList& 
Dataset::files() const
{
  if (not m_files.empty()) return m_files;

  bool hdf5 = this->exists("h5");

  bool smd = this->exists("smd");

  // get directory name where to look for files
  std::string dir = this->dirName();
  if (smd) {
    dir += "/smalldata";
  }
  if (not fs::is_directory(dir)) {
    throw DatasetDirError(ERR_LOC, dir);
  }

  // scan all files in directory, find matching ones
  std::map<unsigned, unsigned> filesPerRun;
  for (fs::directory_iterator fiter(dir); fiter != fs::directory_iterator(); ++ fiter) {

    const fs::path& path = fiter->path();
    const fs::path& basename = path.filename();
    
    for (IData::Dataset::Runs::const_iterator ritr = m_runs.begin(); ritr != m_runs.end(); ++ ritr) {
      for (unsigned run = ritr->first; run <= ritr->second; ++ run) {
        
        // make file name regex 
        std::string reStr;
        if (hdf5) {
          reStr = boost::str(boost::format("%1%-r0*%2%(-.*)?[.]h5") % experiment() % run);
        } else if (smd) {
          reStr = boost::str(boost::format("e%1%-r0*%2%-s[0-9]+-c[0-9]+[.]smd[.]xtc") % expID() % run);          
        } else {
          reStr = boost::str(boost::format("e%1%-r0*%2%-s[0-9]+-c[0-9]+[.]xtc") % expID() % run);          
        }
        boost::regex re(reStr);

        // name should match and we only take regular files
        if (boost::regex_match(basename.string(), re) and fiter->status().type() == fs::regular_file) {
          MsgLog(logger, debug, "found matching file: " << path);
          m_files.push_back(path.string());
          ++ filesPerRun[run];
        }
      }
      
    }
  }

  // Check file count per run, issue warning for runs without files
  for (IData::Dataset::Runs::const_iterator ritr = m_runs.begin(); ritr != m_runs.end(); ++ ritr) {
    // only check runs specified explicitly, not ranges
    if (ritr->first == ritr->second) {
      if (filesPerRun[ritr->first] == 0) {
        MsgLog(logger, warning, "no input files found for run #" << ritr->first);
      }
    }
  }

  return m_files;
}

void 
Dataset::parseXtcFileName(std::string path)
{
  m_key2val["xtc"];

  // leave only basename
  std::string::size_type p = path.rfind('/');
  if (p != std::string::npos) path.erase(0, p+1);
  
  // drop extension
  p = path.rfind('.');
  if (p != std::string::npos) path.erase(p);

  // split into parts
  std::vector<std::string> parts;
  boost::split(parts, path, boost::is_any_of("-"), boost::token_compress_on);
  
  // need at least 2 pieces - experiment and run number
  if (parts.size() < 2) return;
    
  // first part is expected to be experiment id in format eNNNN
  if (parts[0].empty() or parts[0][0] != 'e') return;
  std::string expid(parts[0], 1);

  // must be all digits, and at least one digit
  if (expid.empty() or not boost::all(expid, boost::is_digit())) return;

  // second part is expected to be run number in format rNNNN
  if (parts[1].empty() or parts[1][0] != 'r') return;
  std::string run(parts[1], 1);

  // must be all digits, and at least one digit
  if (run.empty() or not boost::all(run, boost::is_digit())) return;

  // parse and store these
  if (not ::parseExpName(expid, m_expId, m_instrName, m_expName)) {
    MsgLog(logger, warning, "unrecognized experiment ID: " << expid);
  }
  ::parseRuns(run, m_runs);
}


void 
Dataset::parseHdfFileName(std::string path)
{
  m_key2val["h5"];

  // leave only basename
  std::string::size_type p = path.rfind('/');
  if (p != std::string::npos) path.erase(0, p+1);
  
  // drop extension
  p = path.rfind('.');
  if (p != std::string::npos) path.erase(p);

  // split into parts
  std::vector<std::string> parts;
  boost::split(parts, path, boost::is_any_of("-"), boost::token_compress_on);
  
  // need at least 2 pieces - experiment and run number
  if (parts.size() < 2) return;
    
  // first part is expected to be experiment name
  if (parts[0].empty()) return;
  std::string expname(parts[0]);

  // second part is expected to be run number in format rNNNN
  if (parts[1].empty() or parts[1][0] != 'r') return;
  std::string run(parts[1], 1);

  // must be all digits, and at least one digit
  if (run.empty() or not boost::all(run, boost::is_digit())) return;

  // parse and store these
  if (not ::parseExpName(expname, m_expId, m_instrName, m_expName)) {
    MsgLog(logger, warning, "unrecognized experiment name: " << expname);
  }
  ::parseRuns(run, m_runs);
}

} // namespace IData

namespace {


// parse experiment name
bool parseExpName(const std::string& exp, unsigned& expId, std::string& instrName, std::string& expName)
{
  ExpNameDb::ExpNameDatabase namedb;

  if (boost::all(exp, boost::is_digit())) {

    // all digits, must be experiment id
    unsigned num = boost::lexical_cast<unsigned>(exp);
    std::pair<std::string, std::string> instrExp = namedb.getNames(num);
    if (instrExp.first.empty()) {
      return false;
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
        return false;
      }

      expId = instrExp.second;
      instrName = instrExp.first;
      expName = exp;

    } else {

      // both instrument and experiment name is given
      const std::string instrument(exp, 0, p);
      const std::string experiment(exp, p+1);

      unsigned num = namedb.getID(instrument, experiment);
      if (num == 0) {
        return false;
      }

      expId = num;
      instrName = instrument;
      expName = experiment;

    }


  }

  return true;
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

// parse stream list
void parseStreams(const std::string& str, IData::Dataset::Streams& streams)
{
  streams.clear();

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
      throw IData::StreamRangeSpecException(ERR_LOC, str, ex.what());
    }
    if (end < start) {
      throw IData::StreamRangeSpecException(ERR_LOC, str, "the first number in the range must be less or equal to teh last one");
    }

    streams.push_back(IData::Dataset::Streams::value_type(start, end));

  }

}


// checks to see if the string is a file name
bool 
isFileName(const std::string& str)
{
  std::string::size_type col = str.find(':');
  if (col == std::string::npos) {

    // no colons but an equal sign - should be an option
    if (str.find('=') != std::string::npos) return false;
    
    // no colons and either dots or slashes - must be a file,
    // no colons and no dots, no slashes - assume it's an option
    return str.find_first_of("./") != std::string::npos;
  
  } else {
  
    // there are colons, if they are followed by slash or digits still fine for a file name
    // (expect URL-type names to be supported in the future)
    while (col != std::string::npos) {
      
      if (col == str.size()-1) {
        // last character is column, cannot be file name
        return false;
      }
      
      if (str[col+1] != '/' and not std::isdigit(str[col+1])) {
        // colon followed by something other than / or digit, not a file
        return false;
      }
      
      // move to next one
      col = str.find(':', col+1);
    }
  
    return true;
  }
  
}

}
