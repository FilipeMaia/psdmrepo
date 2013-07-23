//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AppCmdWordWrap...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "AppUtils/AppCmdWordWrap.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sys/ioctl.h>
#include <utility>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

// guess page width from terminal size
int pageWidth()
{
  // assume that either stdin, stdout, or stderr is at terminal, otherwise use 80 chars
  struct winsize w;
  if (ioctl(0, TIOCGWINSZ, &w) < 0 and ioctl(1, TIOCGWINSZ, &w) < 0 and ioctl(2, TIOCGWINSZ, &w) < 0) {
    w.ws_col = 80;
  }
  return w.ws_col;
}

// strip leading/trailing white spaces
std::string
strip(const std::string& line)
{
  std::string res;
  // strip leading/trailing blanks, tabs, and newlines
  std::string::size_type p0 = line.find_first_not_of(" \t\n");
  if (p0 == std::string::npos) {
    // it's completely empty/blank
    return res;
  }
  std::string::size_type p1 = line.find_last_not_of(" \t\n");
  return line.substr(p0, p1-p0+1);
}

// split line
std::pair<std::string, std::string>
split2(const std::string& text, unsigned pageWidth)
{
  std::pair<std::string, std::string> res;

  std::string remaining = ::strip(text);

  // if there is a newline within pageWidth, split at it
  std::string::size_type p = remaining.find('\n');
  if (p < pageWidth) {
    res.first = ::strip(remaining.substr(0, p));
    res.second = ::strip(remaining.substr(p+1));
    return res;
  }

  // p0 will point to beginning of the current word optionally
  // preceded by few blank chars
  std::string::size_type p0 = 0;
  while (true) {

    // p1 will point past the current word
    std::string::size_type p1 = remaining.find_first_not_of(" \t\n", p0);
    if (p1 == std::string::npos) {
      p1 = remaining.size();
    } else {
      p1 = remaining.find_first_of(" \t\n", p1);
      if (p1 == std::string::npos) p1 = remaining.size();
    }

    if (p1 >= pageWidth) {

      // need to split now, decide where to split
      if (p1 == pageWidth or p0 == 0) p0 = p1;

      res.first = remaining.substr(0, p0);
      if (p0 < remaining.size()) res.second = remaining.substr(p0);
      break;

    } else if (p1 == remaining.size()) {

      // grab whole remaining stuff
      res.first = remaining;
      break;

    }

    // move to next word
    p0 = p1;

  }

  return res;
}

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace AppUtils {

// Constructor determines page width
AppCmdWordWrap::AppCmdWordWrap(int pageWidth)
  : m_pageWidth(pageWidth)
{
  if (m_pageWidth <= 0) {
    static int pageWidth = ::pageWidth();
    m_pageWidth = pageWidth;
  }
}

/**
 *  Split input string into the set of lines so that each line is no
 *  longer than specified width. If width is negative then the width
 *  determined by constructor is used. Splitting is done on spaces
 *  and tabs, newline characters cause unconditional split.
 */
std::vector<std::string>
AppCmdWordWrap::wrap(const std::string& text, int pageWidth) const
{
  if (pageWidth <= 0) pageWidth = m_pageWidth;

  std::pair<std::string, std::string> pair(std::string(), text);
  std::vector<std::string> lines;
  do {
    pair = ::split2(pair.second, pageWidth);
    lines.push_back(pair.first);
  } while (not pair.second.empty());

  return lines;
}

} // namespace AppUtils
