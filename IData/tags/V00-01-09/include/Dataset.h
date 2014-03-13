#ifndef IDATA_DATASET_H
#define IDATA_DATASET_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Dataset.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <map>
#include <vector>
#include <utility>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace IData {

/// @addtogroup IData

/**
 *  @ingroup IData
 *
 *  @brief C++ class representing dataset concept.
 *
 *  Dataset is defined currently as a collection of one or more runs
 *  from the same experiment. The minimum information that dataset
 *  needs to contain is experiment name/number and run numbers.
 *  In addition to that it provides additional options for locating
 *  data files or selecting specific format (HDF5 vs XTC) of data.
 *
 *  Some option (like experiment name) can be specified at the
 *  application-wide basis by using static methods of this class.
 *  Dataset instances can override those global options by providing
 *  their own values in a constructor.
 *
 *  Constructor of Dataset class takes either a file name or a string 
 *  representation of the dataset which is a list of options separated 
 *  by colon (:) characters:
 *
 *    @code
 *    option[:option[:option...]]
 *    @endcode
 *
 *  Each @c option can be a key-value pair separated by equal sign
 *  or just a key without value:
 *
 *    @code
 *    key1=value1:key2=value2:key3:key4
 *    @endcode
 *
 *  Here is the set of common key names and meaning of their corresponding
 *  values:
 *
 *  @li @b exp - specifies experiment, corresponding value can be either
 *      experiment number (exp=197), experiment name (exp=cxi59712), or
 *      instrument and experiment names separated by slash (exp=CXI/cxi59712)
 *  @li @b run - specifies set of runs belonging to dataset which is a
 *      comma-separated list of run ranges, range can be a single run number
 *      or two number separated by dash (run=12,15-18,25)
 *  @li @b xtc - selects XTC files as input (which is default), no value for this key
 *  @li @b h5 - selects HDF5 files as input, no value for this key
 *  @li @b live - selects live XTC files as input, no value for this key
 *  @li @b dir - specifies directory containing input files, by default
 *      files are located in the standard experiment directories under
 *      /reg/d/psdm directory
 *  @li @b one-stream - this option works with XTC input only, if the option
 *      is given a value (number) then it specifies stream number to read,
 *      without value arbitrary single stream will be selected. If option
 *      is not specified then data from all stream are read.
 *  @li @b stream - specifies set of streams belonging to dataset which is a
 *      comma-separated list of stream ranges, range can be a single stream number
 *      or two number separated by dash (stream=0,2-4,11)
 *
 *  If the same key appears multiple times in the input string the latter
 *  values for this key override earlier values.
 *
 *  If the string passed to a constructor looks like a file name then dataset 
 *  constructor tries to guess various pieces of information from the file name
 *  itself. To look like a file name the string should either:
 *  - do not contain colon character and contain at least one dot
 *  - if colon character is in the string it follows by slash or digit (for 
 *    something like root://host:port/path)
 *
 *  <hr>
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Dataset  {
public:

  typedef std::pair<unsigned, unsigned> RunRange;
  typedef std::vector<RunRange> Runs;
  typedef std::pair<unsigned, unsigned> StreamRange;
  typedef std::vector<StreamRange> Streams;
  typedef std::vector<std::string> NameList;

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
  static void setAppExpName(const std::string& exp);

  /**
   *  @brief Sets default application-wide option.
   *
   *  Sets default application-wide value for an option. Individual datasets can override
   *  application-wide values. The key "run" is ignored by this method, warning
   *  message is printed. With key "exp" this is equivalent to calling setAppExpName(value).
   *
   *  @param[in] key   Key name
   *  @param[in] value New application-wide value for this key
   *
   *  @throw ExpNameException thrown if key is "exp" and specified experiment name is not found
   */
  static void setDefOption(const std::string& key, const std::string& value);

  /// Default constructor makes empty dataset
  Dataset();

  /**
   *  @brief Make dataset instance
   *
   *  Constructor takes string representation of a dataset as described in
   *  class-level documentation. Options specified in the string override
   *  default application-wide options.
   *
   *  @param[in] ds  String representation of dataset.
   *
   *  @throw ExpNameException thrown if specified name is not found
   *  @throw RunNumberException thrown if specified run list has incorrect format.
   */
  Dataset(const std::string& ds);

  // Destructor
  ~Dataset();

  /**
   *  @brief Returns true if the key is defined.
   *
   *  Key may be defined by either constructor or with a default
   *  application-wide option.
   *
   *  @param[in] key  Key name
   */
  bool exists(const std::string& key) const;

  /**
   *  @brief Returns value for given key or empty string.
   *
   *  @param[in] key  Key name
   */
  const std::string& value(const std::string& key) const;

  /// Returns experiment ID or 0 if it has not been defined.
  unsigned expID() const;

  /// Returns instrument name or empty string if it has not been defined.
  const std::string& instrument() const;

  /// Returns experiment name or empty string if it has not been defined.
  const std::string& experiment() const;

  /// Returns set of run numbers
  const Runs& runs() const;

  /// Returns set of stream numbers
  const Streams& streams() const;

  /// Returns true if dataset was specified as a single file name
  bool isFile() const { return m_isFile; }
  
  /// Return the directory name for files, if "dir" option is specified 
  /// the it is returned, otherwise some default lcoation for experiment 
  /// files is returned.
  std::string dirName() const;

  /// Return the list of file names for this dataset
  const NameList& files() const;
  
protected:

  // parse XTC file name
  void parseXtcFileName(std::string path);
  
  // parse HDF file name
  void parseHdfFileName(std::string path);
  
private:

  typedef std::map<std::string, std::string> Key2Val;

  bool m_isFile;             ///< True if dataset is a file name 
  Key2Val m_key2val;         ///< Mapping of keys to values
  Runs m_runs;               ///< Set of runs
  Streams m_streams;         ///< Set of streams
  unsigned m_expId;          ///< Experiment ID
  std::string m_instrName;   ///< Instrument name
  std::string m_expName;     ///< Experiment name
  mutable NameList m_files;  ///< List of file names for this dataset

  static Key2Val s_key2val;         ///< Application-wide options
  static unsigned s_expId;          ///< Application-wide experiment ID
  static std::string s_instrName;   ///< Application-wide instrument name
  static std::string s_expName;     ///< Application-wide experiment name

};

} // namespace IData

#endif // IDATA_DATASET_H
