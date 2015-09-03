<?php

require_once 'dataportal/dataportal.inc.php' ;
require_once 'shiftmgr/shiftmgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;
use ShiftMgr\Utils ;

function export_shifts2excel ($shifts, $user) {

    // Sort shifts by the begin time in the descending order

    function cmp ($a, $b) {
        $a_begin_sec = $a->begin_time()->sec ;
        $b_begin_sec = $b->begin_time()->sec ;
        if ($a_begin_sec == $b_begin_sec) {
            return 0;
        }
        return ($a_begin_sec < $b_begin_sec) ? -1 : 1 ;
    }
    usort($shifts, 'cmp') ;
    rsort($shifts) ;

    require_once 'PHPExcel.php' ;

    $file = "LCLS_Shift_Report_".(LusiTime::now()->sec).".xlsx";
    $path = "/tmp/".$file;

    $title = 'Title' ;

    $objPHPExcel = new PHPExcel() ;

    $objPHPExcel->getProperties()->setCreator($user['gecos'])
                                 ->setLastModifiedBy($user['gecos'])
                                 ->setTitle("Office 2007 XLSX Document")
                                 ->setSubject("LCLS Instruement Shifts")
                                 ->setDescription("Test document for Office 2007 XLSX, generated from PCDS Instrument Shifts Database")
                                 ->setKeywords("office 2007 openxml lcls instrument shift")
                                 ->setCategory("Shifts") ;
    $objPHPExcel->setActiveSheetIndex(0) ;
    $objPHPExcel->getActiveSheet()->getHeaderFooter()->setOddHeader('&14 &B'.$title.', &D &T') ;

    foreach( array('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U') as $k => $cell ) {
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
    $area2cell = array (
        'FEL'  => 'H' ,
        'BMLN' => 'I' ,
        'CTRL' => 'J' ,
        'DAQ'  => 'K' ,
        'LASR' => 'L' ,
        'TIME' => 'M' ,
        'HALL' => 'N' ,
        'OTHR' => 'O'
    ) ;
    $allocation2cell = array (
        'tuning'    => array('descr' => 'Tuning',           'cell' => 'P') ,
        'alignment' => array('descr' => 'Alignment',        'cell' => 'Q') ,
        'daq'       => array('descr' => 'Data Taking',      'cell' => 'R') ,
        'access'    => array('descr' => 'Hutch Access',     'cell' => 'S') ,
        'machine'   => array('descr' => 'Machine Downtime', 'cell' => 'T') ,
        'other'     => array('descr' => 'Other',            'cell' => 'U')
    ) ;

    $row = 1;

    $objPHPExcel->getActiveSheet()->setCellValue('A'.$row, title2bold('Type')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('B'.$row, title2bold('Instr')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('C'.$row, title2bold('Begin')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('D'.$row, title2bold('End')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('E'.$row, title2bold('Duration')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('F'.$row, title2bold('Stopper')) ;
    $objPHPExcel->getActiveSheet()->setCellValue('G'.$row, title2bold('Door open')) ;
    foreach ($area2cell as $area_name => $cell) {
        $objPHPExcel->getActiveSheet()->setCellValue($cell.$row, title2bold($area_name)) ;
    }
    foreach ($allocation2cell as $allocation_name => $cell_descr) {
        $objPHPExcel->getActiveSheet()->setCellValue($cell_descr['cell'].$row, title2bold($cell_descr['descr'])) ;
    }
    foreach( array('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U') as $k => $cell ) {
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
    foreach( $shifts as $shift ) {

        foreach( array('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U') as $k => $cell) {
            set_cell_borders($objPHPExcel, $cell.$row) ;
        }

        $begin_time = $shift->begin_time() ;
        $end_time   = $shift->end_time() ;

        $objPHPExcel->getActiveSheet()->setCellValue('A'.$row, $shift->type()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('B'.$row, $shift->instr_name()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('C'.$row, $begin_time->toStringDay().' '.$begin_time->toStringHM()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('D'.$row, $end_time->toStringDay().' '.$end_time->toStringHM()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('E'.$row, $shift->duration()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('F'.$row, $shift->stopper()) ;
        $objPHPExcel->getActiveSheet()->setCellValue('G'.$row, $shift->door_open()) ;
        foreach ($shift->areas() as $area) {
            $objPHPExcel->getActiveSheet()->setCellValue($area2cell[$area->name()].$row, $area->downtime_min()) ;
        }
        foreach ($shift->allocations() as $allocation) {
            $objPHPExcel->getActiveSheet()->setCellValue($allocation2cell[$allocation->name()]['cell'].$row, $allocation->duration_min()) ;
        }
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


/**
 * The Web Service to export shifts into a file based on the specified search criteria:
 * 
 *   <range> [<begin>] [<end>] [<stopper>] [<door>] [<lcls>] [<daq>] [<instruments>] [<types>]
 */
\DataPortal\Service::run_handler('GET', function($SVC) {
    export_shifts2excel (
        Utils::query_shifts($SVC) ,
        $SVC->regdb()->find_user_account (
            $SVC->authdb()->authName()
        )
    ) ;
}) ;

?>
