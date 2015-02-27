#ifndef PIXMAP_H
#define PIXMAP_H

#include <QWidget>

namespace PSQt {

class Pixmap : public QWidget
{
  Q_OBJECT  

  public:
    Pixmap(QWidget *parent = 0);

  protected:
    void paintEvent(QPaintEvent *event);
};

} // namespace PSQt 

#endif // PIXMAP_H
