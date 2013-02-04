#ifndef O2OTRANSLATOR_CVTOPTIONS_H
#define O2OTRANSLATOR_CVTOPTIONS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CvtOptions.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Type.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/// @addtogroup O2OTranslator

/**
 *  @ingroup O2OTranslator
 *
 *  @brief Collection of options passed to converter instances.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class CvtOptions  {
public:

  /**
   *  @brief Constructor takes values for each option
   *
   *  @param[in]
   *  @param[in] chunkSize   HDF5 chunk size in bytes
   *  @param[in] compLevel     Compression level
   *  @param[in] fillMissing If true then missing data is "stored" and mask dataset is created
   *  @param[in] storeDamage If true then special dataset for damage bitmask is created
   */
  CvtOptions(hsize_t chunkSize, int compLevel, bool fillMissing, bool storeDamage)
    : m_chunkSize(chunkSize)
    , m_compLevel(compLevel)
    , m_fillMissing(fillMissing)
    , m_storeDamage(storeDamage)
  {}

  /// Returns HDF5 chunk size in bytes
  hsize_t chunkSize() const { return m_chunkSize; }

  /// Returns compression level
  int compLevel() const { return m_compLevel; }

  /// Returns flag for treating missing data
  bool fillMissing() const { return m_fillMissing; }

  /// Returns flag for creating damage dataset
  bool storeDamage() const { return m_storeDamage; }

protected:

private:

  hsize_t m_chunkSize;  ///< HDF5 chunk size in bytes
  int m_compLevel;      ///< Compression level
  bool m_fillMissing;   ///< if true then missing data is "stored" and mask dataset is created
  bool m_storeDamage;   ///< if true then special dataset for damage bitmask is created

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CVTOPTIONS_H
