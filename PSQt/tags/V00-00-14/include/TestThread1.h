//--------------------------

#ifndef TESTTHREAD1_H
#define TESTTHREAD1_H

//#include <Qt>
//#include <QtGui>
//#include <QtCore>
//#include <QApplication>
#include <QThread>

namespace PSQt {

/**
 *  @ingroup PSQt
 * 
 *  @brief Tread-worker, inherits from QThread
 * 
 *  @code
 *  @endcode
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see 
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

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




