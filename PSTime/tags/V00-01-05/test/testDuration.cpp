/* Test of time string parsification methods in class Time */

// #include "PSTime/Time.h"
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

  //-------------

  printf ( "\n\n\nTest Duration *duration0 = new Duration();\n");
       Duration *duration0 = new Duration();
                 duration0 -> Print();

	//------------------------------------
	cout << endl << separationLine << endl;

  time_t sec = 1*(3600*24*364) + 2*(3600*24) + 3*(3600) + 4*60 + 5;
  time_t nsec = 123456789;
  printf ( "\n\n\nTest Duration *duration1 = new Duration(%d,%d);\n", (int)sec, (int)nsec );
  Duration *duration1 = new Duration(sec,nsec);
            duration1 -> Print();

	//------------------------------------
	cout << endl << separationLine << endl;


  printf ( "\n\n\nTest Duration *duration3 = new Duration(1,0,3,0,5,0);\n");
  Duration *duration3 = new Duration(1,0,3,0,5,0);
            duration3 -> Print();

	//------------------------------------
	cout << endl << separationLine << endl;

  printf ( "\n\n\nTest Duration *duration4 = new Duration(0,2,0,4,0,6);\n");
  Duration *duration4 = new Duration(0,2,0,4,0,6);
            duration4 -> Print();

	//------------------------------------
	cout << endl << separationLine << endl;

  printf ( "\n\n\nTest Duration duration5 = *duration3 + *duration4;\n");
  Duration  duration5 = *duration3 + *duration4;
            duration5. Print();

	//------------------------------------
	cout << endl << separationLine << endl;

  //-------------  
  return 0;
  //-------------
}
