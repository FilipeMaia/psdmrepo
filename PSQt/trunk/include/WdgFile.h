#ifndef WDGFILE_H
#define WDGFILE_H

//#include <QVBox>
//#include <QWidget>
//#include <QLabel>
//#include <QSlider>
//#include <QPushButton>
#include <QString>

#include <Qt>
#include <QtGui>
#include <QtCore>

namespace PSQt {

class WdgFile : public QWidget
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
    WdgFile( QWidget *parent = 0, 
             const std::string& but_title=std::string("File:"), 
             const std::string& path=std::string("/reg/neh/home1/dubrovin/LCLS/pubs/reflective-geometry.png"),
             const std::string& search_fmt=std::string("*.data *.png \n *") );

    //const std::string& path=std::string("/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2014-05-15/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/2-end.data") );
    //const std::string& path=std::string("/reg/d/psdm/CXI/cxitut13/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data") );

    void resizeEvent(QResizeEvent *event = 0) ;
    void closeEvent (QCloseEvent  *event = 0) ;

 public slots:
    void onButFile() ;
    void onEdiFile() ;
    void testSignalString(const std::string& fname) ;

 signals :
    void fileNameIsChanged(const std::string&  fname) ;
    void valueChanged(int value) ;

 private :
    std::string  m_path;
    std::string  m_search_fmt;
    //QLabel*      m_lab_fname;
    QFrame*      m_frame;
    QLineEdit*   m_edi_file;
    QPushButton* m_but_file;

    void showTips() ;
    void setFrame() ;
    bool fileExists(const std::string& fname) ;
    bool setNewFileName(const std::string& fname) ;
};

} // namespace PSQt

#endif // WDGFILE_H
