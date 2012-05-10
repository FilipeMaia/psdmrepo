#ifndef ALLINONE_H
#define ALLINONE_H

//#include <QVBox>
//#include <QWidget>
//#include <QLabel>
//#include <QSlider>
//#include <QPushButton>

#include <Qt>
#include <QtGui>
#include <QtCore>

namespace PSQt {

class MyWidget : public QWidget
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
    MyWidget( QWidget *parent = 0 );

    void setFrame() ;
    void resizeEvent(QResizeEvent *event = 0) ;
    void closeEvent (QCloseEvent  *event = 0) ;
    void moveEvent  (QMoveEvent   *event = 0) ;
    void mousePressEvent (QMouseEvent *event = 0) ;

 public slots:
    void onButton() ;
    void onRadio1() ;
    void onRadio2() ;
    void onTextEdit() ;
    void onLineEdit() ;
    void onComboBox() ;
    void onCheckBox() ;
    void onSpinBox() ;
    void onCustomButton() ;
    void onLineWindow() ;
    void onDonutWindow() ;

 private:
    QFrame*    m_frame;
    QLineEdit* m_line_edit;
};

} // namespace PSQt

#endif
