#ifndef PSXTCINPUT_EXCEPTIONS_H
#define PSXTCINPUT_EXCEPTIONS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Exceptions.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <cerrno>
#include <boost/lexical_cast.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "ErrSvc/Issue.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSXtcInput {

/**
 *  Exception classes
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class Exception : public ErrSvc::Issue {
public:

  // Constructor
  Exception ( const ErrSvc::Context& ctx, 
              const std::string& className, 
              const std::string& what ) ;

};

// thrown for empty file list
class EmptyFileList : public Exception {
public:

  EmptyFileList(const ErrSvc::Context& ctx)
    : Exception( ctx, "EmptyFileList", "No input file names specified" ) {}

};

// thrown when live mode requested in indexing
class NoLiveIndex : public Exception {
public:

  NoLiveIndex(const ErrSvc::Context& ctx)
    : Exception( ctx, "NoLiveIndex", "Live mode not supported with indexing" ) {}

};

// thrown when run is not found in the dataset
class RunNotInDataset : public Exception {
public:

  RunNotInDataset(const ErrSvc::Context& ctx)
    : Exception( ctx, "RunNotInDataset", "Run not found in dataset" ) {}

};

// thrown when run is not found in the dataset
class CalibCycleNotFound : public Exception {
public:

  CalibCycleNotFound(const ErrSvc::Context& ctx)
    : Exception( ctx, "CalibCycleNotFound", "Calib Cycle not found" ) {}

};

class EpicsDataNotFound : public Exception {
public:

  EpicsDataNotFound(const ErrSvc::Context& ctx)
    : Exception( ctx, "EpicsDataNotFound", "Epics data not found" ) {}

};

// thrown when xtc file open fails
class XTCNotFound : public Exception {
public:

  XTCNotFound(const ErrSvc::Context& ctx)
    : Exception( ctx, "XTCNotFound", "Unable to open XTC file" ) {}

};

// thrown when xtc file open fails
class IndexSeekFailed : public Exception {
public:

  IndexSeekFailed(const ErrSvc::Context& ctx)
    : Exception( ctx, "IndexseekFailed", "Indexing seek failed" ) {}

};

// thrown for empty file
class EmptyInput : public Exception {
public:

  EmptyInput(const ErrSvc::Context& ctx)
    : Exception( ctx, "EmptyInput", "XTC file(s) is empty" ) {}

};

// thrown for unexpected input
class UnexpectedInput : public Exception {
public:

  UnexpectedInput(const ErrSvc::Context& ctx)
    : Exception( ctx, "UnexpectedInput", "Number of datagrams received from source is not expected" ) {}

};

/// Exception class which extracts error info from errno.
class ExceptionErrno : public Exception {
public:

  /// Constructor takes the reason for an exception
  ExceptionErrno ( const ErrSvc::Context& ctx, const std::string& what ) ;

};

} // namespace PSXtcInput

#endif // PSXTCINPUT_EXCEPTIONS_H
