#ifndef DONUT_H
#define DONUT_H

#include <QWidget>

namespace PSQt {

class Donut : public QWidget
{
  Q_OBJECT  

  public:
    Donut(QWidget *parent = 0);

  protected:
    void paintEvent(QPaintEvent *event);
};

} // namespace PSQt 

#endif // DONUT_H
