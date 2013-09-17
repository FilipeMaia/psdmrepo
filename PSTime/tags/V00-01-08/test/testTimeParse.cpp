/* Test of time string parsification methods in class Time */

#include "PSTime/TimeFormat.h"

#include <stdio.h>
#include <time.h>

#include <iostream>
#include <string>

using namespace std;
using namespace PSTime;

int main ()
{
  //-------------
  char separationLine[]="----------------------------------";

  string time_stamp_human1="2010-11-21 10:58:51.123456789-08:00";
  string time_stamp_human2="2010-11-21 10:58:51.123456Z";
  string time_stamp_human3="2010-11-21 10:58:51-08:00";
  string time_stamp_human4="2010-11-21 10:58:51Z";

  string time_stamp_basic1="20101121T105851.12346-0800";
  string time_stamp_basic2="20101121T105851.12346Z";
  string time_stamp_basic3="20101121T105851-0800";
  string time_stamp_basic4="20101121T105851Z";

  string time_stamp_mixed1="20101121 105851.55+01";
  string time_stamp_mixed2="20101121T10:58:51-0115";
  string time_stamp_mixed3="2010-11-21T105851Z";
  string time_stamp_mixed4="2010-11-21T105851+0145";

  //-------------

  cout << "\n\n\nTest Time test_time1 = new Time();\n";
  Time test_time1 ;
  cout << test_time1 << endl;

  //------------------------------------
  cout << endl << separationLine << endl
       << "Test parsification of human-readable time stamps" << endl;

  cout << endl << "Parse t-stamp: " << time_stamp_human1 << endl;
  test_time1 = TimeFormat::parseTime(       time_stamp_human1 );
  cout << test_time1 << endl;

  cout << endl << "Parse t-stamp: " << time_stamp_human2 << endl;
  test_time1 = TimeFormat::parseTime(       time_stamp_human2 );
  cout << test_time1 << endl;

  cout << endl << "Parse t-stamp: " << time_stamp_human3 << endl;
  test_time1 = TimeFormat::parseTime(       time_stamp_human3 );
  cout << test_time1 << endl;

  cout << endl << "Parse t-stamp: " << time_stamp_human4 << endl;
  test_time1 = TimeFormat::parseTime(       time_stamp_human4 );
  cout << test_time1 << endl;

  //------------------------------------
  cout << endl << separationLine << endl
       << "Test parsification of basic-format time stamps" << endl;

  cout << endl << "Parse t-stamp: " << time_stamp_basic1 << endl;
  test_time1 = TimeFormat::parseTime(       time_stamp_basic1 );
  cout << test_time1 << endl;

  cout << endl << "Parse t-stamp: " << time_stamp_basic2 << endl;
  test_time1 = TimeFormat::parseTime(       time_stamp_basic2 );
  cout << test_time1 << endl;

  cout << endl << "Parse t-stamp: " << time_stamp_basic3 << endl;
  test_time1 = TimeFormat::parseTime(       time_stamp_basic3 );
  cout << test_time1 << endl;

  cout << endl << "Parse t-stamp: " << time_stamp_basic4 << endl;
  test_time1 = TimeFormat::parseTime(       time_stamp_basic4 );
  cout << test_time1 << endl;

  //------------------------------------
  cout << endl << separationLine << endl
       << "Test parsification of mixed-format time stamps" << endl;

  cout << endl << "Parse t-stamp: " << time_stamp_mixed1 << endl;
  test_time1 = TimeFormat::parseTime(       time_stamp_mixed1 );
  cout << test_time1 << endl;

  cout << endl << "Parse t-stamp: " << time_stamp_mixed2 << endl;
  test_time1 = TimeFormat::parseTime(       time_stamp_mixed2 );
  cout << test_time1 << endl;

  cout << endl << "Parse t-stamp: " << time_stamp_mixed3 << endl;
  test_time1 = TimeFormat::parseTime(       time_stamp_mixed3 );
  cout << test_time1 << endl;

  cout << endl << "Parse t-stamp: " << time_stamp_mixed4 << endl;
  test_time1 = TimeFormat::parseTime(       time_stamp_mixed4 );
  cout << test_time1 << endl;

  //------------------------------------
  cout << endl << separationLine << endl
       << "Test parsification of other mixed-format time stamps" << endl;

  cout << endl << "Parse t-stamp: " << "20101121 105851" << endl;
  test_time1 = TimeFormat::parseTime(       "20101121 105851" );
  cout << test_time1 << endl;

  cout << endl << "Parse t-stamp: " << "20101121T105851.33" << endl;
  test_time1 = TimeFormat::parseTime(       "20101121T105851.33" );
  cout << test_time1 << endl;

  cout << endl << "Parse t-stamp: " << "20101121 105851.33+01" << endl;
  test_time1 = TimeFormat::parseTime(       "20101121 105851.33+01" );
  cout << test_time1 << endl;

  cout << endl << separationLine << endl;


  //------------------------------------
  cout << endl << separationLine << endl
       << "Test constructor with string parsification" << endl;

  cout << endl << "Parse t-stamp: " << time_stamp_human1 << endl;
  Time time_pars1(               time_stamp_human1);
  cout << time_pars1 << endl;

  cout << endl << "Parse t-stamp: " << time_stamp_basic1 << endl;
  Time time_pars2(               time_stamp_basic1);
  cout << time_pars2 << endl;

  //------------------------------------
  cout << endl << separationLine << endl;

  //-------------  
  return 0;
  //-------------
}
