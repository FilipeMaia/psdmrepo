// Compilation: http://linuxhelp.blogspot.com/2006/01/creating-and-compiling-qt-projects-on.html
//
//
#include <QApplication>
#include <QLabel>
//#include <QtGui>
 
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QLabel label("Hello, world!");
    label.show();
    return app.exec();
}
