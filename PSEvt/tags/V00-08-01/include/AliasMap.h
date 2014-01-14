#ifndef PSEVT_ALIASMAP_H
#define PSEVT_ALIASMAP_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AliasMap.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <map>
#include <string>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Src.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSEvt {

/// @addtogroup PSEvt

/**
 *  @ingroup PSEvt
 *
 *  @brief Implementation of the alias map used by proxy dictionaries.
 *
 *  Alias map is a bi-directional mapping between alias name and corresponding
 *  Pds::Src instance.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class AliasMap {
public:

  /// Default constructor makes empty alias map
  AliasMap();

  // Destructor
  ~AliasMap();

  /// Add one more alias to the map
  void add(const std::string& alias, const Pds::Src& src);

  /// remove all aliases
  void clear();

  /**
   *  @brief Find matching Src for given alias name.
   *
   *  If specified alias name does not exist in the map then
   *  default-constructed instance of Src will be returned.
   *
   *  @param[in] alias    Alias name
   *  @return  Src instance
   */
  Pds::Src src(const std::string& alias) const;

  /**
   *  @brief Find matching alias name for given Src.
   *
   *  If specified Src does not exist in the map then
   *  empty string will be returned.
   *
   *  @param[in] src   Src instance
   *  @return  Alias string (possibly empty)
   */
  std::string alias(const Pds::Src& src) const;

protected:

private:

  // Special compare functor which compares Src instances, we need to ignore
  // process ID part of Src.
  struct SrcCmp {
    bool operator()(const Pds::Src& lhs, const Pds::Src& rhs) const;
  };
  
  std::map<std::string, Pds::Src> m_alias2src;         ///< Mapping from alias name to Src
  std::map<Pds::Src, std::string, SrcCmp> m_src2alias; ///< Mapping from Src to alias name

};

} // namespace PSEvt

#endif // PSEVT_ALIASMAP_H
