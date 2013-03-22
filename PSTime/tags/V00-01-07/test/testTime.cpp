
/* Test of time methods in class Time */

#include "PSTime/Time.h"

#include <iostream>
#include <time.h>


using namespace std;
using namespace PSTime;

int main ()
{
  cout << "\n\nTest implementation of the class PSTime\n";
  //-------------

  cout << "\n\n\nTest  Time *UTCtime1 = new Time();\n";   
  Time *UTCtime1 = new Time();
  cout << *UTCtime1 << endl;

  //-------------

//  cout << "\n\n\nTest getPSTZoneTimeOffsetSec() = " << Time::getPSTZoneTimeOffsetSec() 
//       << " sec = " << UTCtime1 -> getPSTZoneTimeOffsetSec()/3600. << " h.\n";

  //-------------

  cout << "\n\n\nTest Time *UTCtime2 = new Time(*UTCtime1);\n";   
  Time *UTCtime2 = new Time(*UTCtime1);
  cout << *UTCtime2 << endl;

  //-------------

  cout << "\n\n\nTest Time *UTCtime3 = new Time(UTCtime1->getUTCSec()+10,UTCtime1->getUTCNsec()+10);\n";
  Time *UTCtime3 = new Time(UTCtime1->sec()+10,UTCtime1->nsec()+10);
  cout << *UTCtime3 << endl;

  //-------------

  cout << "\n\n\nTest Time *UTCtime4 = new Time(year,month,day,hour,min,sec,nsec,zone)";
  cout << "\n  for parameters (2010,10,8,17,04,43,123456,Time::UTC);\n";
  Time *UTCtime4 = new Time(2010,10,8,17,04,43,123456,Time::UTC);
  cout << *UTCtime4 << endl;

  cout << "\n  for parameters (2010,10,8,17,04,43,123456,Time::Local);\n";
  Time *UTCtime5 = new Time(2010,10,8,17,04,43,123456,Time::Local);
  cout << *UTCtime5 << endl;

  cout << "\n  for parameters (2010,10,8,17,04,43,123456,Time::Local);\n";
  Time *UTCtime6 = new Time(2010,10,8,17,04,43,123456,Time::Local);
  cout << *UTCtime6 << endl;

  //-------------
  struct timespec currentHRTime;
  int gettimeStatus = 0;
      gettimeStatus = clock_gettime( CLOCK_REALTIME, &currentHRTime );
  cout << "\n\n\nTest   Time *UTCtime7 = new Time(currentHRTime);\n";
  Time *UTCtime7 = new Time(currentHRTime);
  cout << *UTCtime7 << endl;

  //-------------
  time_t     local=time(NULL);                // current local time
  tm         locTime=*localtime(&local);      // convert local to local, store as tm
  tm         utcTime=*gmtime(&local);         // convert local to GMT, store as tm

  cout << "\n\n\nTest   Time *UTCtime8 = new Time(utcTime,Time::UTC);\n";   
  Time *UTCtime8 = new Time(utcTime,Time::UTC);
  cout << *UTCtime8 << endl;

  //-------------

//  cout << "\n\nTest UTCtime8->strZoneHuman(Time::UTC)   : " << UTCtime8->strZoneHuman(Time::UTC) << endl ;
//  cout << "\n\nTest UTCtime8->strZoneHuman(Time::Local) : " << UTCtime8->strZoneHuman(Time::Local) << endl ;   
//
//  cout << "\n\nTest UTCtime8->strZoneBasic(Time::UTC)   : " << UTCtime8->strZoneBasic(Time::UTC) << endl ;   
//  cout << "\n\nTest UTCtime8->strZoneBasic(Time::Local) : " << UTCtime8->strZoneBasic(Time::Local) << endl ;
//
//  //-------------
//
//  cout << "\n\nTest UTCtime8->strDateHuman(Time::UTC)   : " << UTCtime8->strDateHuman(Time::UTC) << endl ; 
//  cout << "\n\nTest UTCtime8->strDateHuman(Time::Local) : " << UTCtime8->strDateHuman(Time::Local) << endl ;
//
//  cout << "\n\nTest UTCtime8->strDateBasic(Time::UTC)   : " << UTCtime8->strDateBasic(Time::UTC) << endl ;
//  cout << "\n\nTest UTCtime8->strDateBasic(Time::Local) : " << UTCtime8->strDateBasic(Time::Local) << endl ;
//
//  //-------------
//
//  cout << "\n\nTest UTCtime8->strTimeHuman(Time::UTC)   : " << UTCtime8->strTimeHuman(Time::UTC) << endl ;
//  cout << "\n\nTest UTCtime8->strTimeHuman(Time::Local) : " << UTCtime8->strTimeHuman(Time::Local) << endl ;
//
//  cout << "\n\nTest UTCtime8->strTimeBasic(Time::UTC)   : " << UTCtime8->strTimeBasic(Time::UTC) << endl ;
//  cout << "\n\nTest UTCtime8->strTimeBasic(Time::Local) : " << UTCtime8->strTimeBasic(Time::Local) << endl ;

//  cout << "\n\nTest UTCtime8->strDateTimeFreeFormat()   : " << UTCtime8->strDateTimeFreeFormat() << endl ;
//  cout << "\n\nTest UTCtime8->strDateTimeFreeFormat(<DateTime format>,Time::Local)   : " <<
//                           UTCtime8->strDateTimeFreeFormat("%B %D %Z %Y%m%d %H:%M:%S",Time::Local) << endl ;

  //-------------

  //Time UTCtime9 = Time::getTimeNow();
  
  time_t sec  = 1287712731;
  time_t nsec = 123456789;
  
  Time *UTCtime9 = new Time(sec,nsec);
  
  cout << *UTCtime9 << endl;
//  cout << "\nTest UTCtime9->strNsec(0)  : " << UTCtime9->strNsec(0) << endl ;
//  cout << "\nTest UTCtime9->strNsec(1)  : " << UTCtime9->strNsec(1) << endl ;
//  cout << "\nTest UTCtime9->strNsec(6)  : " << UTCtime9->strNsec(6) << endl ;
//  cout << "\nTest UTCtime9->strNsec(9)  : " << UTCtime9->strNsec(9) << endl ;
//  cout << "\nTest UTCtime9->strNsec(15) : " << UTCtime9->strNsec(15) << endl ;
  
  //-------------
  
  cout << "\n\nTest UTCtime9->asStringCompact(Time::Local)   : " << UTCtime9->asStringCompact(Time::Local) << endl ;
  cout << "\n\nTest UTCtime9->asStringCompact(Time::Local,5) : " << UTCtime9->asStringCompact(Time::Local,5) << endl ;
  cout << "\n\nTest UTCtime9->asStringCompact(Time::Local,0) : " << UTCtime9->asStringCompact(Time::Local,0) << endl ;
  
  cout << "\n\nTest UTCtime9->asString(Time::Local)   : " << UTCtime9->asString(Time::Local) << endl ;
  cout << "\n\nTest UTCtime9->asString(Time::Local,3) : " << UTCtime9->asString(Time::Local,3) << endl ;
  cout << "\n\nTest UTCtime9->asString(Time::Local,0) : " << UTCtime9->asString(Time::Local,0) << endl ;
  
  cout << "\n\nTest UTCtime9->asStringFormat()   : " << UTCtime9->asStringFormat() << endl ; 
  cout << "\n\nTest UTCtime9->asStringFormat(<DateTime format>,Time::Local)   : " <<
                     UTCtime9->asStringFormat("%Y%m%d %H:%M:%S%.4f%z",Time::Local) << endl ;
  

  //-------------  
  return 0;
  //-------------
}
