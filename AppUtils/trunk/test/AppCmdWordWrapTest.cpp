//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test suite case for the AppCmdLine & friends.
//
// Environment:
//	Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//	Andy Salnikov		originator
//
// Copyright Information:
//	Copyright (C) 2003	SLAC
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------
#include <string>
#include <algorithm>
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdWordWrap.h"

using namespace AppUtils;
using namespace std;


#define BOOST_TEST_MODULE AppCmdWordWrapTest
#include <boost/test/included/unit_test.hpp>

namespace {

  bool compare(const char* expect[], size_t n_expect, const vector<string>& result)
  {
    cout << "Expected: " << n_expect << " strings\n";
    for (size_t i = 0; i != n_expect; ++ i) {
      cout << "    " << i << ": \"" << expect[i] << "\"\n";
    }

    cout << "Received: " << result.size() << " strings\n";
    for (size_t i = 0; i != result.size(); ++ i) {
      cout << "    " << i << ": \"" << result[i] << "\"\n";
    }

    if (n_expect != result.size()) {
      cout << "*** Fail *** - result size is different\n";
      return false;
    }

    if (not std::equal(result.begin(), result.end(), expect)) {
      cout << "*** Fail *** - strings are different\n";
      return false;
    }

    cout << "+++ OK +++\n";
    return true;
  }

}

// ==============================================================

BOOST_AUTO_TEST_CASE(cmdlww_test_set_page_width)
{
  AppCmdWordWrap ww(100);

  BOOST_CHECK_EQUAL(ww.pageWidth(), 100);
}

// ==============================================================

BOOST_AUTO_TEST_CASE(cmdlww_test_1)
{
  AppCmdWordWrap ww(10);

  const char longLine[] = "abc abc abc abcde abc";
  const char* lines[] = {
      "abc abc",
      "abc abcde",
      "abc",
  };
  size_t nlines = sizeof lines / sizeof lines[0];

  BOOST_CHECK(::compare(lines, nlines, ww.wrap(longLine)));
}

// ==============================================================

BOOST_AUTO_TEST_CASE(cmdlww_test_spaces)
{
  AppCmdWordWrap ww(10);

  const char longLine[] = "  abc abc ab    \t  \t abcde  abc     ";
  const char* lines[] = {
      "abc abc ab",
      "abcde  abc",
  };
  size_t nlines = sizeof lines / sizeof lines[0];

  BOOST_CHECK(::compare(lines, nlines, ww.wrap(longLine)));
}

// ==============================================================

BOOST_AUTO_TEST_CASE(cmdlww_test_newlines)
{
  AppCmdWordWrap ww(10);

  const char longLine[] = "  abc\nabc   \n ab abcde\nabc     ";
  const char* lines[] = {
      "abc",
      "abc",
      "ab abcde",
      "abc",
  };
  size_t nlines = sizeof lines / sizeof lines[0];

  BOOST_CHECK(::compare(lines, nlines, ww.wrap(longLine)));
}

// ==============================================================

BOOST_AUTO_TEST_CASE(cmdlww_test_longword)
{
  AppCmdWordWrap ww(10);

  const char longLine[] = " 01234567890123456789  01234567890123456789 0123 ";
  const char* lines[] = {
      "01234567890123456789",
      "01234567890123456789",
      "0123",
  };
  size_t nlines = sizeof lines / sizeof lines[0];

  BOOST_CHECK(::compare(lines, nlines, ww.wrap(longLine)));
}

// ==============================================================

BOOST_AUTO_TEST_CASE(cmdlww_test_longword_nl)
{
  AppCmdWordWrap ww(1);

  const char longLine[] = " 01234567890123456789\n01234567890123456789 \n0123 ";
  const char* lines[] = {
      "01234567890123456789",
      "01234567890123456789",
      "0123",
  };
  size_t nlines = sizeof lines / sizeof lines[0];

  BOOST_CHECK(::compare(lines, nlines, ww.wrap(longLine)));
}

// ==============================================================
