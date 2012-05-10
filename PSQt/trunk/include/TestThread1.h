//--------------------------

#ifndef TESTTHREAD1_H
#define TESTTHREAD1_H

//#include <Qt>
//#include <QtGui>
//#include <QtCore>
//#include <QApplication>
#include <QThread>

namespace PSQt {

//--------------------------

class TestThread1 : public QThread
{
  Q_OBJECT

 public:
    TestThread1(QObject *parent = 0);

 protected:
    void run();

 private:
    uint m_thread;
};

//--------------------------

} // namespace PSQt

#endif // TESTTHREAD1_H

//--------------------------




