//---------------------------------------------------------------------
// File and Version Information:
//   $Id$
//
// Author: Mikhail S. Dubrovin
//---------------------------------------------------------------------


#include "PSQt/ThreadTimer.h"
#include <iostream>
using  namespace std;

namespace PSQt {

//--------------------------

  ThreadTimer::ThreadTimer(QObject* parent, unsigned dt_sec, bool pbits)
  : QThread(parent)
  , m_count(0)
  , m_dt_sec(dt_sec)
  , m_pbits(pbits)
{
  static uint thread_counter = 0; thread_counter++;
  m_thread_number = thread_counter;
}

//--------------------------

void ThreadTimer::run()
{
  do {

    if (m_pbits & 1) cout << " Thread:" << m_thread_number << " loop =" << ++m_count << endl;
    sleep(m_dt_sec); 

  } while (true);
}

//--------------------------
} // namespace PSQt
//--------------------------
