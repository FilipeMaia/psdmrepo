#ifndef BRUSHES_H
#define BRUSHES_H

#include <QWidget>

namespace PSQt {

class Brushes : public QWidget
{
  Q_OBJECT  

  public:
    Brushes(QWidget *parent = 0);

  protected:
    void paintEvent(QPaintEvent *event);

};

} // namespace PSQt 

#endif // BRUSHES_H
