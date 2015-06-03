//--------------------------------------------------------------------------
// File and Version Information:
//     $Id$
//
// Description:
//     Class LoggerNameWithMpiRank
//
// Author List:
//     David Schneider
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Translator/LoggerNameWithMpiRank.h"

//-----------------
// C/C++ Headers --
//-----------------
#include "openmpi/mpi.h"
#include "boost/lexical_cast.hpp"

//             ----------------------------------------
//             -- Public Function Member Definitions --
//             ----------------------------------------

LoggerNameWithMpiRank::operator const std::string &() {
  if (m_firstCall) {
    setName();
    m_firstCall = false;
  }
  return m_loggerName;
}  

void LoggerNameWithMpiRank::setName() {
  int initialized;
  MPI_Initialized(&initialized);
  if (not initialized) return;

  int worldRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  m_loggerName += " mpiRnk=";
  m_loggerName += boost::lexical_cast<std::string>(worldRank);
}
