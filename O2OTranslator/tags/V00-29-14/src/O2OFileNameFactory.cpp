//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OFileNameFactory...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/O2OFileNameFactory.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  std::string seq2value ( unsigned int seq, int w ) {
    std::ostringstream str ;
    str.fill('0') ;
    str.width(w) ;
    str << seq ;
    return str.str() ;
  }
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
O2OFileNameFactory::O2OFileNameFactory ( const std::string& fileNameTemplate )
  : m_fileNameTemplate(fileNameTemplate)
  , m_key2value()
{
}

//--------------
// Destructor --
//--------------
O2OFileNameFactory::~O2OFileNameFactory ()
{
}

/// add substitution keyword and its value
void
O2OFileNameFactory::addKeyword ( const std::string& key, const std::string& value )
{
  m_key2value.insert ( Key2Value::value_type( key, value) ) ;
}

/// generate full path name
std::string
O2OFileNameFactory::makePath(int seq) const
{
  std::string result = m_fileNameTemplate ;

  std::string::size_type pos = 0 ;
  // find first '{'
  while ( ( pos = result.find( '{', pos ) ) != std::string::npos ) {

    // find matching '}'
    std::string::size_type pos2 = result.find( '}', pos ) ;
    if ( pos2 == std::string::npos ) {
      // no closing braces left, nothing left to do
      break ;
    }

    // get the keyword
    std::string key ( result, pos+1, pos2-pos-1 ) ;

    if ( key.empty() ) {
      // empty key - just remove braces and continue
      key.erase ( pos, pos2-pos+1 ) ;
      continue ;
    }

    // find the value for this key
    std::string value ;
    if ( key.size() > 2 and key[0] == 's' ) {
      if (seq == Family) {
        // test for 'seq#'
        if ( key == "seq" ) {
          value = "%d" ;
        } else if ( key == "seq2" ) {
          value = "%02d" ;
        } else if ( key == "seq3" ) {
          value = "%03d" ;
        } else if ( key == "seq4" ) {
          value = "%04d" ;
        } else if ( key == "seq5" ) {
          value = "%05d" ;
        } else if ( key == "seq6" ) {
          value = "%06d" ;
        } else if ( key == "seq7" ) {
          value = "%07d" ;
        } else if ( key == "seq8" ) {
          value = "%08d" ;
        } else if ( key == "seq9" ) {
          value = "%09d" ;
        } else if ( key == "seq10" ) {
          value = "%010d" ;
        }
      } else if (seq == FamilyPattern) {
        // test for 'seq#'
        if ( key == "seq" ) {
          value = "[0-9]+" ;
        } else if ( key == "seq2" ) {
          value = "[0-9]{2}" ;
        } else if ( key == "seq3" ) {
          value = "[0-9]{3}" ;
        } else if ( key == "seq4" ) {
          value = "[0-9]{4}" ;
        } else if ( key == "seq5" ) {
          value = "[0-9]{5}" ;
        } else if ( key == "seq6" ) {
          value = "[0-9]{6}" ;
        } else if ( key == "seq7" ) {
          value = "[0-9]{7}" ;
        } else if ( key == "seq8" ) {
          value = "[0-9]{8}" ;
        } else if ( key == "seq9" ) {
          value = "[0-9]{9}" ;
        } else if ( key == "seq10" ) {
          value = "[0-9]{10}" ;
        }
      } else {
        // test for 'seq#'
        if ( key == "seq" ) {
          value = seq2value( seq, 0 ) ;
        } else if ( key == "seq2" ) {
          value = seq2value( seq, 2 ) ;
        } else if ( key == "seq3" ) {
          value = seq2value( seq, 3 ) ;
        } else if ( key == "seq4" ) {
          value = seq2value( seq, 4 ) ;
        } else if ( key == "seq5" ) {
          value = seq2value( seq, 5 ) ;
        } else if ( key == "seq6" ) {
          value = seq2value( seq, 6 ) ;
        } else if ( key == "seq7" ) {
          value = seq2value( seq, 7 ) ;
        } else if ( key == "seq8" ) {
          value = seq2value( seq, 8 ) ;
        } else if ( key == "seq9" ) {
          value = seq2value( seq, 9 ) ;
        } else if ( key == "seq10" ) {
          value = seq2value( seq, 10 ) ;
        }
      }
    }

    if ( value.empty() ) {
      // find it in the map
      Key2Value::const_iterator it = m_key2value.find ( key ) ;
      if ( it != m_key2value.end() ) value = it->second ;
    }

    if ( not value.empty() ) {
      // we have a value to substitute
      result.replace ( pos, pos2-pos+1, value ) ;
    } else {
      // leave it as it is, move to the next
      pos = pos2+1 ;
    }

  }

  return result ;
}

} // namespace O2OTranslator
