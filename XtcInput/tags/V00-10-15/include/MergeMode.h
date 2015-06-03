#ifndef XTCINPUT_MERGEMODE_H
#define XTCINPUT_MERGEMODE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MergeMode.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iosfwd>
#include <string>

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

namespace XtcInput {

/// @addtogroup XtcInput

/**
 *  @ingroup XtcInput
 *
 *  @brief Enum which defines merge modes supported by iterator classes
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

enum MergeMode {
  MergeFileName,      ///< streams and chunks are determined from file names
  MergeOneStream,     ///< All files come from one stream, chunked
  MergeNoChunking,    ///< Single file per stream, no chunking
};

/**
 *  @brief Make merge mode from string
 *  
 *  @throw InvalidMergeMode Thrown if string does not match the names 
 *    of enum constants
 */
MergeMode mergeMode(const std::string& str);


/// Insertion operator for enum values
std::ostream&
operator<<(std::ostream& out, MergeMode mode);


} // namespace XtcInput

#endif // XTCINPUT_MERGEMODE_H
