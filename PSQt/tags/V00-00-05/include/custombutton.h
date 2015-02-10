#ifndef CUSTOMBUTTON_H
#define CUSTOMBUTTON_H

//Custom button class
//Qt-articles.blogspot.com

#include <Qt>
#include <QtGui>
#include <QtCore>

namespace PSQt {

class CustomButton : public QPushButton {
    Q_OBJECT
 
public:
    CustomButton(QWidget *parent = 0);
    ~CustomButton();
 
public:
    //QString FirstName, MiddleName, LastName;
    QImage SimpleIcon;
 
protected:
    void paintEvent(QPaintEvent *); 
};

} // namespace PSQt

#endif // CUSTOMBUTTON_H
