/* Test of duration string parsification methods in the class Duration */

#include "PSTime/Duration.h"

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

  string str_dur1 = "P1Y2M3DT05H6M7S";
  string str_dur2 = "P01Y02M03D";
  string str_dur3 = "PT05H06M07S";
  string str_dur4 = "PT07.00345600S";

  int status;

  cout << separationLine << endl;
  //-------------

  Duration *dur1 = new Duration();
            dur1 -> Print();
  cout << endl << "Parse : " << str_dur1  << endl;
  status = Duration::parseStringToDuration(str_dur1, *dur1);
  dur1 -> Print();
  cout << separationLine << endl; 

  //-------------

  Duration *dur2 = new Duration();
  cout << endl << "Parse : " << str_dur2  << endl;
  status = Duration::parseStringToDuration(str_dur2, *dur2);
  dur2 -> Print();
  cout << separationLine << endl;

  //-------------

  Duration *dur3 = new Duration();
  cout << endl << "Parse : " << str_dur3  << endl;
  status = Duration::parseStringToDuration(str_dur3, *dur3);
  dur3 -> Print();
  cout << separationLine << endl;

  //-------------

  Duration *dur4 = new Duration();
  cout << endl << "Parse : " << str_dur4  << endl;
  status = Duration::parseStringToDuration(str_dur4, *dur4);
  dur4 -> Print();
  cout << separationLine << endl;

  //-------------
}
