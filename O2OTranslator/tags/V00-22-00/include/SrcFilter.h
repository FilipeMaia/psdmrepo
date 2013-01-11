#ifndef O2OTRANSLATOR_SRCFILTER_H
#define O2OTRANSLATOR_SRCFILTER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class SrcFilter.
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
# include "pdsdata/xtc/Src.hh"

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
 *  @brief Class defining filter for different type of sources.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class SrcFilter  {
public:

  enum SourceType { Any = -1, BLD = Pds::Level::Reporter };

  /// Default constructor means allow anything
  SrcFilter () : m_source(Any), m_allow(true) {}

  /**
   *   Factory for method which returns filter accepting specified source.
   *   allow(Any) accepts all data sources, allow(BLD) only accepts BLD.
   */
  static SrcFilter allow(SourceType source) { return SrcFilter(source, true); }

  /**
   *   Factory for method which returns filter rejecting specified source
   *   unless argument is Any which means accept anyhting.
   *   deny(Any) accepts all data sources, allow(BLD) only accepts BLD.
   */
  static SrcFilter deny(SourceType source) { return SrcFilter(source, false); }

  /// Test source against filter, return true if source is accepted
  bool operator()(const Pds::Src& src) const {
    if (m_allow) {
      return m_source == Any or int(src.level()) == int(m_source);
    } else {
      return m_source == Any or int(src.level()) != int(m_source);
    }
  }

protected:

  SrcFilter (SourceType source, bool allow) : m_source(source), m_allow(allow) {}

private:

  // Data members
  SourceType m_source;
  bool m_allow;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_SRCFILTER_H
