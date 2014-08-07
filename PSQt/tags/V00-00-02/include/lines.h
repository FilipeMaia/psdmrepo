#ifndef LINES_H
#define LINES_H

//#include <Qt>
//#include <QtGui>
//#include <QtCore>

#include <QWidget>
#include <QFrame>

namespace PSQt {

class Lines : public QWidget
{
  Q_OBJECT  

  public:
    Lines(QWidget *parent = 0);

  protected:
    void paintEvent(QPaintEvent *event);
    void setFrame(); 

  private:
    QFrame*    m_frame;
};

} // namespace PSQt 

#endif // LINES_H
