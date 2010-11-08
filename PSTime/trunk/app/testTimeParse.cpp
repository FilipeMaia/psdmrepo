/* Test of time string parsification methods in class Time */

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

void time_stamp_sparser(string &tstamp)
{
  size_t pos=0; 

  printf("\n tstamp.data()   = %s\n",tstamp.data()   );
  printf("\n tstamp.length() = %d\n",(int)tstamp.length() );
  printf("\n tstamp.size()   = %d\n",(int)tstamp.size()   );

    cout << "string::npos = " << int(string::npos) << endl;

  pos=tstamp.find('.');
  if (pos!=string::npos) cout << "Period found at: " << int(pos) << endl;

  pos=tstamp.find(' ');
  if (pos!=string::npos) cout << "Space found at: " << int(pos) << endl;

  char char_date[16];
  char char_time[16];
  size_t lend=tstamp.copy(char_date,10,0);    char_date[lend] = '\0';
  size_t lent=tstamp.copy(char_time,8,pos+1); char_time[lent] = '\0';
  cout << "separated date = " << char_date << " has length " << lend << endl; 
  cout << "separated time = " << char_time << " has length " << lent << endl; 

  pos=tstamp.find('T');
  if (pos!=string::npos) cout << "T found at: " << int(pos) << endl;
  else                   cout << "T is not found " << endl;

}



int main ()
{
  //-------------
  char strLine[]="----------------------------------";
  printf ("\n\nTest various standard time methods\n");
  printf ("%s\n",strLine);
  time_t seconds;
  seconds = time(NULL);
  printf ("\nTest seconds = time(NULL);\n %ld UTC seconds since January 1, 1970", seconds);

  struct tm * timeinfo;
  timeinfo = localtime ( &seconds );

  print_struct_tm(timeinfo);

  //-------------
  printf ("\n\nTest strptime(...)\n");
  printf ("%s\n",strLine);

  struct tm * tm_human;
  struct tm * tm_basic;

  char date_human[]="2010-10-25";
  char date_basic[]="20101025";

  char time_human[]="17:24:51";
  char time_basic[]="172451";

  if(strptime(time_human,"%H:%M:%S",tm_human) == NULL)  printf("\nstrptime failed\n");
  else print_struct_tm(tm_human);

  if(strptime(time_basic,"%H%M%S",tm_basic) == NULL)  printf("\nstrptime failed\n");
  else print_struct_tm(tm_basic);

  if(strptime(date_basic,"%Y%m%d",tm_basic) == NULL)  printf("\nstrptime failed\n");
  else print_struct_tm(tm_basic);

  if(strptime(date_human,"%Y-%m-%d",tm_human) == NULL)  printf("\nstrptime failed\n");
  else print_struct_tm(tm_basic);

  string str_test="ABCZ";
  printf ( "\n\n\nTest string.find %d;\n",(int)str_test.find('Z'));


  string time_stamp_human1="2010-10-21 10:58:51.123456789-08:00";
  string time_stamp_human2="2010-10-21 10:58:51.123456Z";
  string time_stamp_human3="2010-10-21 10:58:51-08:00";
  string time_stamp_human4="2010-10-21 10:58:51Z";

  string time_stamp_basic1="20101021T105851.12346-0800";
  string time_stamp_basic2="20101021T105851.12346Z";
  string time_stamp_basic3="20101021T105851-0800";
  string time_stamp_basic4="20101021T105851Z";

  //-------------
  //  printf("\nsizeof time_stamp_human1[] = %d\n",(int)sizeof(time_stamp_human1) );
  //  time_stamp_sparser( time_stamp_human1 ); 

  printf ( "\n\n\nTest Time *test_time1 = new Time();\n");
  Time *test_time1 = new Time();
        test_time1 -> Print();


    int status = Time::parseTimeStamp(time_stamp_human1, *test_time1);
        test_time1 -> Print();

	status = Time::parseTimeStamp(time_stamp_human2, *test_time1);
        test_time1 -> Print();

	status = Time::parseTimeStamp(time_stamp_human3, *test_time1);
        test_time1 -> Print();

	status = Time::parseTimeStamp(time_stamp_human4, *test_time1);
        test_time1 -> Print();

        status = Time::parseTimeStamp("20101021T105851", *test_time1);
        test_time1 -> Print();

        status = Time::parseTimeStamp("20101021T105851.33", *test_time1);
        test_time1 -> Print();

        status = Time::parseTimeStamp("20101021 105851.33+01", *test_time1);
        test_time1 -> Print();



  //-------------

  //-------------
  //-------------
  //-------------
  //-------------

  //-------------  
  return 0;
  //-------------
}
