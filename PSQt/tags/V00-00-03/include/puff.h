#ifndef PUFF_H
#define PUFF_H

#include <QWidget>

namespace PSQt {

class Puff : public QWidget
{
  Q_OBJECT  

  public:
    Puff(QWidget *parent = 0);

  protected:
    void paintEvent(QPaintEvent *event);
    void timerEvent(QTimerEvent *event);

  private:
    int x;
    qreal opacity;
    int timerId;

};

} // namespace PSQt 

#endif // PUFF_H
