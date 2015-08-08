<?php

/**
  * Search equipment on behalf of a Web service an return an Excel spreadsheet
  *
  * See a list of parameters supported by the search in a description of the utility
  * function. The only additional parameter recognized here is:
  * 
  *   <format>
  * 
  * @see  \Irep\IrepUtils::find_equipment
  */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'irep/irep.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

function export_equipment2excel ($equipment, $user) {


    require_once 'PHPExcel.php' ;

    $file = "PCDS_Equipment_Report_".(LusiTime::now()->sec).".xlsx";
    $path = "/tmp/".$file;

    $title = 'Title' ;

    $objPHPExcel = new PHPExcel() ;

    $objPHPExcel->getProperties()->setCreator        ($user['gecos'])
                                 ->setLastModifiedBy ($user['gecos'])
                                 ->setTitle          ("Office 2007 XLSX Document")
                                 ->setSubject        ("PCDS Inventory and Repair Database")
                                 ->setDescription    ("Test document for Office 2007 XLSX, generated from PCDS Inventory and Repair Database")
                                 ->setKeywords       ("office 2007 openxml pcds inventory repair equipment")
                                 ->setCategory       ("Equipment") ;
    $objPHPExcel->setActiveSheetIndex(0) ;
    $objPHPExcel->getActiveSheet()->getHeaderFooter()->setOddHeader('&14 &B'.$title.', &D &T') ;

    foreach( array('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O') as $k => $cell ) {
        $objPHPExcel->getActiveSheet()->getColumnDimension($cell)->setAutoSize(true) ;
    }

    function set_cell_borders ($e, $c) {
        $e->getActiveSheet()->getStyle($c)->applyFromArray(array (
            'borders'  => array(
                'top'  => array('style' => PHPExcel_Style_Border::BORDER_THIN),
                'left' => array('style' => PHPExcel_Style_Border::BORDER_THIN))
        )) ;
    }
    function title2bold ($title) {
        $objRichText = new PHPExcel_RichText() ;
        $objRichText->createTextRun($title)->getFont()->setBold(true) ;
        return $objRichText;
    }


    $row = 1;

    $objPHPExcel->getActiveSheet()->setCellValue('A'.$row, title2bold('Status')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('B'.$row, title2bold('Sub-status')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('C'.$row, title2bold('Manufacturer')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('D'.$row, title2bold('Model')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('E'.$row, title2bold('Serial #')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('F'.$row, title2bold('SLAC ID')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('G'.$row, title2bold('PC #')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('H'.$row, title2bold('Location')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('I'.$row, title2bold('Room')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('J'.$row, title2bold('Rack')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('K'.$row, title2bold('Elevation')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('L'.$row, title2bold('Custodian')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('M'.$row, title2bold('Modified')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('N'.$row, title2bold('By user')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('O'.$row, title2bold('Notes')) ;


    foreach( array('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O') as $k => $cell ) {
        set_cell_borders($objPHPExcel, $cell.$row) ;
    }
    $objPHPExcel->getActiveSheet()->getStyle('A3:S'.$row)->applyFromArray( array(
        'font'      => array('bold'       => true),
        'alignment' => array('horizontal' => PHPExcel_Style_Alignment::HORIZONTAL_LEFT),
        'borders'   => array('top'        => array('style' => PHPExcel_Style_Border::BORDER_THIN)),
        'fill'      => array('type'       => PHPExcel_Style_Fill::FILL_GRADIENT_LINEAR,
                             'rotation'   => 90,
                             'startcolor' => array('argb' => 'FFA0A0A0'),
                             'endcolor'   => array('argb' => 'FFFFFFFF' )))) ;

    $objPHPExcel->getActiveSheet()->getPageSetup()->setRowsToRepeatAtTopByStartAndEnd(1, 3) ;

    $row = 2;
    foreach ($equipment as $e) {

        foreach( array('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O') as $k => $cell) {
            set_cell_borders($objPHPExcel, $cell.$row) ;
        }

        $modified_time = $e->modified_time() ;
        $modified_uid  = $e->modified_uid () ;

        $objPHPExcel->getActiveSheet()->setCellValue('A'.$row, $e->status()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('B'.$row, $e->status2()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('C'.$row, $e->manufacturer()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('D'.$row, $e->model()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('E'.$row, $e->serial()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('F'.$row, $e->slacid()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('G'.$row, $e->pc()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('H'.$row, $e->location()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('I'.$row, $e->room()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('J'.$row, $e->rack()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('K'.$row, $e->elevation()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('L'.$row, $e->custodian()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('M'.$row, $modified_time ? $modified_time->toStringDay().' '.$modified_time->toStringHM() : '') ;
        $objPHPExcel->getActiveSheet()->setCellValue('N'.$row, $modified_uid  ? $modified_uid : '') ;
        $objPHPExcel->getActiveSheet()->setCellValue('O'.$row, $e->description()) ;


        $row += 1 ;
    }

    $objPHPExcel->setActiveSheetIndex(0) ;

    $objWriter = PHPExcel_IOFactory::createWriter($objPHPExcel, 'Excel2007') ;
    $objWriter->save($path) ;

    header("Content-type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet") ;
    header('Content-Disposition: attachment; filename='.$file) ;

    ob_clean() ;
    flush() ;
    readfile($path) ;

    exit() ;
}

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    $format = $SVC->required_str('format') ;
    if ($format === 'excel') {
        export_equipment2excel (
            \Irep\IrepUtils::find_equipment($SVC) ,
            $SVC->regdb()->find_user_account (
                $SVC->authdb()->authName()
            )) ;
    }
    $SVC->abort("unsupported export format: '{$format}'") ;
}) ;

?>
