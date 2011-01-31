/* Test of basic time access and string parsification methods for the class Time */

#include "PSTime/Time.h"

#include <stdio.h>
#include <time.h>

#include <iostream>
#include <string>

using namespace std;
using namespace PSTime;

void print_struct_tm(struct tm * timeinfo)
{
  printf ( "\nasctime (timeinfo)  : %s", asctime(timeinfo));
  printf ( "\nTest timeinfo->tm_sec, timeinfo->tm_min, ...:\n");  

  printf ( "sec  = %2d  mday = %9d  year = %9d\n", timeinfo->tm_sec,  timeinfo->tm_mday,  timeinfo->tm_year );
  printf ( "min  = %2d  mon  = %9d  wday = %9d\n", timeinfo->tm_min,  timeinfo->tm_mon,   timeinfo->tm_wday );
  printf ( "hour = %2d  isdst= %9d  yday = %9d\n", timeinfo->tm_hour, timeinfo->tm_isdst, timeinfo->tm_yday );
}


int main ()
{
  //-------------
  char strLine[]="----------------------------------";
  printf ("\n\nTest various standard time methods\n");

  time_t seconds;
  seconds = time(NULL);
  printf ("\n\n\nTest seconds = time(NULL);\n %ld UTC seconds since January 1, 1970", seconds);

  printf ("\n\n\nTest struct tm *timeinfo = localtime ( &seconds );\n");  
  struct tm * timeinfo;
  timeinfo = localtime ( &seconds );

  printf ("\nTest timeinfo->tm_sec, timeinfo->tm_min, ...:\n");  
  printf ( "sec  = %d\n", timeinfo->tm_sec  );
  printf ( "min  = %d\n", timeinfo->tm_min  );
  printf ( "hour = %d\n", timeinfo->tm_hour );
  printf ( "mday = %d\n", timeinfo->tm_mday );
  printf ( "mon  = %d\n", timeinfo->tm_mon  );
  printf ( "year = %d\n", timeinfo->tm_year );
  printf ( "wday = %d\n", timeinfo->tm_wday );
  printf ( "yday = %d\n", timeinfo->tm_yday );
  printf ( "isdst= %d\n", timeinfo->tm_isdst);
  printf ( "\n asctime (timeinfo)  : %s", asctime (timeinfo) );
  printf ( " or ctime (&seconds) : %s\n", ctime (&seconds) );

  char str[256];
  size_t maxsize = 255;
  int len = strftime(str, maxsize, "Allows to constract a local time stamp though w/o nsec :\n %Y-%m-%d %H:%M:%S %Z", timeinfo); 
  printf ( "strftime(...) gives string : %s of length = %d\n",str,len);  

  //-------------

  time_t     local=time(NULL);                // current local time
  tm         locTime=*localtime(&local);      // convert local to local, store as tm
  tm         utcTime=*gmtime(&local);         // convert local to GMT, store as tm
  time_t     utc=(mktime(&utcTime));          // convert tm to time_t 
  double     diff=difftime(utc,local)/3600;   // difference in hours
  printf ( "curr(s) = %d, utc(s) = %d, diff(h) = %f\n", (int)local, (int)utc, diff); 
  printf ( " locTime.tm_hour = %d\n", locTime.tm_hour); 
  printf ( " utcTime.tm_hour = %d\n", utcTime.tm_hour); 

  //-------------

  struct timespec currentHRTime;
  int gettimeStatus = clock_gettime( CLOCK_REALTIME, &currentHRTime );
  printf ( "\n\n\nTest gettimeStatus = clock_gettime( CLOCK_REALTIME, &currentHRTime ); \n gettimeHRStatus =  %d  currentHRTime.tv_sec = %ld  .tv_nsec = %ld\n", gettimeStatus, currentHRTime.tv_sec, currentHRTime.tv_nsec );  

  //-------------

  printf ("%s\n",strLine);
  seconds = time(NULL);
  printf ("\nTest seconds = time(NULL);\n %ld UTC seconds since January 1, 1970", seconds);

  timeinfo = localtime ( &seconds );

  print_struct_tm(timeinfo);

  //-------------
  printf ("\n\nTest strptime(...)\n");
  printf ("%s\n",strLine);

  struct tm tm_human;
  struct tm tm_basic;

  char date_human[]="2010-10-25";
  char date_basic[]="20101025";

  char time_human[]="17:24:51";
  char time_basic[]="172451";

  if(strptime(time_human,"%H:%M:%S",&tm_human) == NULL)  printf("\nstrptime failed\n");
  else print_struct_tm(&tm_human);

  if(strptime(time_basic,"%H%M%S",&tm_basic) == NULL)  printf("\nstrptime failed\n");
  else print_struct_tm(&tm_basic);

  if(strptime(date_basic,"%Y%m%d",&tm_basic) == NULL)  printf("\nstrptime failed\n");
  else print_struct_tm(&tm_basic);

  if(strptime(date_human,"%Y-%m-%d",&tm_human) == NULL)  printf("\nstrptime failed\n");
  else print_struct_tm(&tm_basic);

  string str_test="ABCZ";
  printf ( "\n\n\nTest string.find %d;\n",(int)str_test.find('Z'));

}
