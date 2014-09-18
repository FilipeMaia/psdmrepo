//--------------------------

#ifndef THREADTIMER_H
#define THREADTIMER_H

#include <QThread>

namespace PSQt {

//--------------------------

class ThreadTimer : public QThread
{
  Q_OBJECT

 public:
  ThreadTimer(QObject *parent=0, unsigned dt_sec=1, bool pbits=0);

 protected:
    void run();

 private:
    unsigned m_thread_number;
    unsigned m_count;
    unsigned m_dt_sec;
    bool     m_pbits;
};

//--------------------------

} // namespace PSQt

#endif // THREADTIMER_H

//--------------------------




