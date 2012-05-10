#ifndef SHAPES_H
#define SHAPES_H

#include <QWidget>

namespace PSQt {

class Shapes : public QWidget
{
  Q_OBJECT  

  public:
    Shapes(QWidget *parent = 0);

  protected:
    void paintEvent(QPaintEvent *event);

};

} // namespace PSQt 

#endif // SHAPES_H
