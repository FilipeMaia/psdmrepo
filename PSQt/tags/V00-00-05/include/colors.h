#ifndef COLORS_H
#define COLORS_H

#include <QWidget>

namespace PSQt {

class Colors : public QWidget
{
  public:
    Colors(QWidget *parent = 0);

  protected:
    void paintEvent(QPaintEvent *event);

};

} // namespace PSQt 

#endif // COLORS_H
