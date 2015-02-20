#ifndef TEXT_H
#define TEXT_H

#include <QWidget>

namespace PSQt {

class Text : public QWidget
{
  Q_OBJECT  

  public:
    Text(QWidget *parent = 0);

  protected:
    void paintEvent(QPaintEvent *event);
};

} // namespace PSQt 

#endif // TEXT_H
