//---------------------------------------------------------------------
// File and Version Information:
//   $Id$
//
// Author: Mikhail S. Dubrovin
//---------------------------------------------------------------------


#include "PSQt/TestThread1.h"
#include <iostream>
using  namespace std;

namespace PSQt {

//--------------------------

TestThread1::TestThread1(QObject* parent)
  : QThread(parent)
{
  static uint counter = 0; counter++;
  m_thread = counter;
}

//--------------------------

void TestThread1::run()
{
  for(int i=0; i<1000000; i++){
    cout << " Thread:" << m_thread << " loop i=" << i << endl;
    sleep(1); // delay in in ms ?
  }  
}

//--------------------------
} // namespace PSQt
//--------------------------
