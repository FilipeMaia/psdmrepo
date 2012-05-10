#ifndef GRADIENT_H
#define GRADIENT_H

#include <QWidget>

namespace PSQt {

class Gradient : public QWidget
{

  public:
    Gradient(QWidget *parent = 0);

  protected:
    void paintEvent(QPaintEvent *event);

};

} // namespace PSQt 

#endif // GRADIENT_H

