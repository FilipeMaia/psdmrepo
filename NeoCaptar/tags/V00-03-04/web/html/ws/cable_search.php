<?php

/**
 * Search and return all known cables in the specified (by its identifier)
 * project.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'neocaptar/neocaptar.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use NeoCaptar\NeoCaptar;
use NeoCaptar\NeoCaptarUtils;
use NeoCaptar\NeoCaptarException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use RegDB\RegDB;
use RegDB\RegDBException;

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

function export_cables2excel($cables,$path,$user) {

    require_once( 'PHPExcel.php' );
  
    $title = count($cables) ? $cables[0]->project()->title() : '';
    $owner = count($cables) ? $cables[0]->project()->owner() : '';

    $objPHPExcel = new PHPExcel();

    $objPHPExcel->getProperties()->setCreator($user['gecos'])
                                 ->setLastModifiedBy($user['gecos'])
                                 ->setTitle("Office 2007 XLSX Neo-CAPTAR Document")
                                 ->setSubject("Cable Project")
                                 ->setDescription("Test document for Office 2007 XLSX, generated from PCDS Neo-CAPTAR Database")
                                 ->setKeywords("office 2007 openxml neocaptar cable")
                                 ->setCategory("Cables");
    $objPHPExcel->setActiveSheetIndex(0);
    $objPHPExcel->getActiveSheet()->getHeaderFooter()->setOddHeader('&14 &B'.$title.', &D &T');

    $objPHPExcel->getActiveSheet()->getColumnDimension('A')->setAutoSize(true);
    $objPHPExcel->getActiveSheet()->getColumnDimension('B')->setAutoSize(true);
    $objPHPExcel->getActiveSheet()->getColumnDimension('D')->setAutoSize(true);
    $objPHPExcel->getActiveSheet()->getColumnDimension('E')->setAutoSize(true);
    $objPHPExcel->getActiveSheet()->getColumnDimension('F')->setAutoSize(true);
    $objPHPExcel->getActiveSheet()->getColumnDimension('G')->setAutoSize(true);
    $objPHPExcel->getActiveSheet()->getColumnDimension('H')->setAutoSize(true);
    $objPHPExcel->getActiveSheet()->getColumnDimension('I')->setAutoSize(true);
    $objPHPExcel->getActiveSheet()->getColumnDimension('J')->setAutoSize(true);
    $objPHPExcel->getActiveSheet()->getColumnDimension('K')->setAutoSize(true);
    $objPHPExcel->getActiveSheet()->getColumnDimension('L')->setAutoSize(true);
    $objPHPExcel->getActiveSheet()->getColumnDimension('M')->setAutoSize(true);
    $objPHPExcel->getActiveSheet()->getColumnDimension('N')->setAutoSize(true);
    $objPHPExcel->getActiveSheet()->getColumnDimension('O')->setAutoSize(true);
    $objPHPExcel->getActiveSheet()->getColumnDimension('P')->setAutoSize(true);
    $objPHPExcel->getActiveSheet()->getColumnDimension('R')->setAutoSize(true);
    $objPHPExcel->getActiveSheet()->getColumnDimension('S')->setAutoSize(true);

    function set_cell_borders($e,$c) {
        $e->getActiveSheet()->getStyle($c)->applyFromArray( array(
            'borders'  => array(
                'top'  => array('style' => PHPExcel_Style_Border::BORDER_THIN),
                'left' => array('style' => PHPExcel_Style_Border::BORDER_THIN))
        ));
    }
    function title2bold($title) {
        $objRichText = new PHPExcel_RichText();
        $objRichText->createTextRun($title)->getFont()->setBold(true);
        return $objRichText;
    }  
    $row = 1;
    $objPHPExcel->getActiveSheet()->setCellValue('A'.$row, title2bold('Status'));
    $objPHPExcel->getActiveSheet()->setCellValue('B'.$row, title2bold('Job #'));
    $objPHPExcel->getActiveSheet()->setCellValue('C'.$row, title2bold('Cable #'));
    $objPHPExcel->getActiveSheet()->setCellValue('D'.$row, title2bold('Device'));
    $objPHPExcel->getActiveSheet()->setCellValue('E'.$row, title2bold('Function'));
    $objPHPExcel->getActiveSheet()->setCellValue('F'.$row, title2bold('Cable Type'));
    $objPHPExcel->getActiveSheet()->setCellValue('G'.$row, title2bold('Length'));
    $objPHPExcel->getActiveSheet()->setCellValue('H'.$row, title2bold('Routing'));
    $objPHPExcel->getActiveSheet()->setCellValue('I'.$row, title2bold('Source/Destination'));
    $objPHPExcel->getActiveSheet()->setCellValue('J'.$row, title2bold('Loc.'));
    $objPHPExcel->getActiveSheet()->setCellValue('K'.$row, title2bold('Rack'));
    $objPHPExcel->getActiveSheet()->setCellValue('L'.$row, title2bold('Ele.'));
    $objPHPExcel->getActiveSheet()->setCellValue('M'.$row, title2bold('Side'));
    $objPHPExcel->getActiveSheet()->setCellValue('N'.$row, title2bold('Slot'));
    $objPHPExcel->getActiveSheet()->setCellValue('O'.$row, title2bold('Conn #'));
    $objPHPExcel->getActiveSheet()->setCellValue('P'.$row, title2bold('Station'));
    $objPHPExcel->getActiveSheet()->setCellValue('Q'.$row, title2bold('Con.Type'));
    $objPHPExcel->getActiveSheet()->setCellValue('R'.$row, title2bold('Pinlist'));
    $objPHPExcel->getActiveSheet()->setCellValue('S'.$row, title2bold('Instr.'));
    foreach( array('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S') as $k=>$v )
        set_cell_borders($objPHPExcel,$v.$row);

    $objPHPExcel->getActiveSheet()->getStyle('A3:S'.$row)->applyFromArray( array(
        'font'      => array('bold'       => true),
        'alignment' => array('horizontal' => PHPExcel_Style_Alignment::HORIZONTAL_LEFT),
        'borders'   => array('top'        => array('style' => PHPExcel_Style_Border::BORDER_THIN)),
        'fill'      => array('type'       => PHPExcel_Style_Fill::FILL_GRADIENT_LINEAR,
                             'rotation'   => 90,
                             'startcolor' => array('argb' => 'FFA0A0A0'),
                             'endcolor'   => array('argb' => 'FFFFFFFF' ))));

    $objPHPExcel->getActiveSheet()->getPageSetup()->setRowsToRepeatAtTopByStartAndEnd(1,3);

    function set_vertical_pair($e,$c,$v1,$v2) {
        $e->getActiveSheet()->getCell($c)->setValue(" {$v1} \n {$v2} ");
        $e->getActiveSheet()->getStyle($c)->getAlignment()->setWrapText(true);
    }
    $row = 2;
    foreach( $cables as $cable ) {

        foreach( array('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S') as $k=>$v )
            set_cell_borders($objPHPExcel,$v.$row);

        $objPHPExcel->getActiveSheet()->setCellValue('A'.$row,$cable->status());
        $objPHPExcel->getActiveSheet()->setCellValue('B'.$row,$cable->project()->job());
        $objPHPExcel->getActiveSheet()->setCellValue('C'.$row,$cable->cable());
        $objPHPExcel->getActiveSheet()->setCellValue('D'.$row,$cable->device());
        $objPHPExcel->getActiveSheet()->setCellValue('E'.$row,$cable->func());
        $objPHPExcel->getActiveSheet()->setCellValue('F'.$row,$cable->cable_type());
        $objPHPExcel->getActiveSheet()->setCellValue('G'.$row,$cable->length());
        $objPHPExcel->getActiveSheet()->setCellValue('H'.$row,$cable->routing());

        set_vertical_pair($objPHPExcel, 'I'.$row, $cable->origin_name(),     $cable->destination_name());
        set_vertical_pair($objPHPExcel, 'J'.$row, $cable->origin_loc(),      $cable->destination_loc());
        set_vertical_pair($objPHPExcel, 'K'.$row, $cable->origin_rack(),     $cable->destination_rack());
        set_vertical_pair($objPHPExcel, 'L'.$row, $cable->origin_ele(),      $cable->destination_ele());
        set_vertical_pair($objPHPExcel, 'M'.$row, $cable->origin_side(),     $cable->destination_side());
        set_vertical_pair($objPHPExcel, 'N'.$row, $cable->origin_slot(),     $cable->destination_slot());
        set_vertical_pair($objPHPExcel, 'O'.$row, $cable->origin_conn(),     $cable->destination_conn());
        set_vertical_pair($objPHPExcel, 'P'.$row, $cable->origin_station(),  $cable->destination_station());
        set_vertical_pair($objPHPExcel, 'Q'.$row, $cable->origin_conntype(), $cable->destination_conntype());
        set_vertical_pair($objPHPExcel, 'R'.$row, $cable->origin_pinlist(),  $cable->destination_pinlist());
        set_vertical_pair($objPHPExcel, 'S'.$row, $cable->origin_instr(),    $cable->destination_instr());

        $row += 1;
    }

    $objPHPExcel->setActiveSheetIndex(0);

    $objWriter = PHPExcel_IOFactory::createWriter($objPHPExcel, 'Excel2007');
    $objWriter->save($path);

    header("Content-type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet") ;
    header('Content-Disposition: attachment; filename='.$path) ;

    ob_clean() ;
    flush() ;
    readfile($path) ;

    exit() ;
}

$export_tools = array(
    'excel' => array('convertor' => 'export_cables2excel','extension' => '.xlsx' )
);

try {
    $authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

    $regdb = RegDB::instance();
    $regdb->begin();

    $uid  = $authdb->authName();
    $user = $regdb->find_user_account($uid);
    if( is_null($user)) NeoCaptarUtils::report_error("no such user: {$uid}");

    // Search option I: exact search
    //
    // Parameters for a simple search based on one of the attributes
    // of a cable (cables).
    //
    // Note that no more that one of those parameters is allowed by the script.
    // A presence of the parameters will be evaluated in an order they are
    // scanned below. Parameters (if specifid) are not allowed to have empty
    // values.
    //
    $id                   = NeoCaptarUtils::get_param_GET('id',                  false);
    $project_id           = NeoCaptarUtils::get_param_GET('project_id',          false);
    $cablenumber          = NeoCaptarUtils::get_param_GET('cablenumber',         false);
    $prefix               = NeoCaptarUtils::get_param_GET('prefix',              false);
    $cablenumber_range_id = NeoCaptarUtils::get_param_GET('cablenumber_range_id',false);
    $jobnumber            = NeoCaptarUtils::get_param_GET('jobnumber',           false);
    $dict_cable_id        = NeoCaptarUtils::get_param_GET('dict_cable_id',       false);
    $dict_connector_id    = NeoCaptarUtils::get_param_GET('dict_connector_id',   false);
    $dict_pinlist_id      = NeoCaptarUtils::get_param_GET('dict_pinlist_id',     false);
    $dict_location_id     = NeoCaptarUtils::get_param_GET('dict_location_id',    false);
    $dict_rack_id         = NeoCaptarUtils::get_param_GET('dict_rack_id',        false);
    $dict_routing_id      = NeoCaptarUtils::get_param_GET('dict_routing_id',     false);
    $dict_instr_id        = NeoCaptarUtils::get_param_GET('dict_instr_id',       false);

    $dict_device_location_id  = NeoCaptarUtils::get_param_GET('dict_device_location_id', false);
    $dict_device_region_id    = NeoCaptarUtils::get_param_GET('dict_device_region_id',   false);
    $dict_device_component_id = NeoCaptarUtils::get_param_GET('dict_device_component_id',false);

    // Search option II: true multi-parametric search
    //
    // Parameters for a complex pattern-based search based on partial
    // values of cable attributes. Parameters are not required. If a parameter
    // is found then it may or may not have a value. If no parameter is found
    // then it's value is assumed to be empty (string).
    //
    function force_empty($str) { return is_null($str) ? '' : $str; }
    $partial_search_params = array(
        'cable'            => force_empty( NeoCaptarUtils::get_param_GET('partial_cable',          false,true)),
        'job'              => force_empty( NeoCaptarUtils::get_param_GET('partial_job',            false,true)),
        'cable_type'       => force_empty( NeoCaptarUtils::get_param_GET('partial_cable_type',     false,true)),
        'routing'          => force_empty( NeoCaptarUtils::get_param_GET('partial_routing',        false,true)),
        'device'           => force_empty( NeoCaptarUtils::get_param_GET('partial_device',         false,true)),
        'func'             => force_empty( NeoCaptarUtils::get_param_GET('partial_func',           false,true)),
        'origin_name'      => force_empty( NeoCaptarUtils::get_param_GET('partial_origin_loc',     false,true)),
        'destination_name' => force_empty( NeoCaptarUtils::get_param_GET('partial_destination_loc',false,true)),
        'partial_or'       => force_empty( NeoCaptarUtils::get_param_GET('partial_or',             false,true)),
    );

    // Output format options:
    //
    // By default if the 'format' parameter is either not present or it's empty
    // cables found as a result of the operation are packaged directly into
    // a JSON object returned to a caller. If the parameter has non-empty
    // values then a list of cables will be exported into a local file of
    // the requested format. A relative URL path to that file will be returned
    // in the JSON object. Then it will be up to a caller to retrieve that
    // file by contacting the corresponding service.
    //
    // The following formats are supported by this script:
    //
    //   'excel' - Microsoft Excel 2007
    //
    $format = force_empty( NeoCaptarUtils::get_param_GET('format',false,true));
    if(($format != '') && !array_key_exists($format, $export_tools))
        NeoCaptarUtils::report_error("format '{$format}' is either not supported or not implemented");

    $cables = array();

    if(!is_null($id)) {

        $cable = $neocaptar->find_cable_by_id($id);
        if( is_null($cable)) NeoCaptarUtils::report_error("cable not found for cable id: {$id}");
        array_push($cables, $cable);

    } else if( !is_null($project_id)) {

        $project = $neocaptar->find_project_by_id($project_id);
        if( is_null($project)) NeoCaptarUtils::report_error("project not found for id: {$project_id}");
        $cables = $project->cables();

    } else if(!is_null($cablenumber)) {

        $cable = $neocaptar->find_cable_by_cablenumber($cablenumber);
        if( is_null($cable)) NeoCaptarUtils::report_error("cable not found for cable number: {$cablenumber}");
        array_push($cables,$cable);

    } else if(!is_null($prefix))                   { $cables = $neocaptar->find_cables_by_prefix                  ($prefix);
    } else if(!is_null($cablenumber_range_id))     { $cables = $neocaptar->find_cables_by_cablenumber_range_id    ($cablenumber_range_id);
    } else if(!is_null($jobnumber))                { $cables = $neocaptar->find_cables_by_jobnumber               ($jobnumber);
    } else if(!is_null($dict_cable_id))            { $cables = $neocaptar->find_cables_by_dict_cable_id           ($dict_cable_id);
    } else if(!is_null($dict_connector_id))        { $cables = $neocaptar->find_cables_by_dict_connector_id       ($dict_connector_id);
    } else if(!is_null($dict_pinlist_id))          { $cables = $neocaptar->find_cables_by_dict_pinlist_id         ($dict_pinlist_id);
    } else if(!is_null($dict_location_id))         { $cables = $neocaptar->find_cables_by_dict_location_id        ($dict_location_id);
    } else if(!is_null($dict_rack_id))             { $cables = $neocaptar->find_cables_by_dict_rack_id            ($dict_rack_id);
    } else if(!is_null($dict_routing_id))          { $cables = $neocaptar->find_cables_by_dict_routing_id         ($dict_routing_id);
    } else if(!is_null($dict_instr_id))            { $cables = $neocaptar->find_cables_by_dict_instr_id           ($dict_instr_id);
    } else if(!is_null($dict_device_location_id))  { $cables = $neocaptar->find_cables_by_dict_device_location_id ($dict_device_location_id);
    } else if(!is_null($dict_device_region_id))    { $cables = $neocaptar->find_cables_by_dict_device_region_id   ($dict_device_region_id);
    } else if(!is_null($dict_device_component_id)) { $cables = $neocaptar->find_cables_by_dict_device_component_id($dict_device_component_id);
    } else                                         { $cables = $neocaptar->search_cables                          ($partial_search_params);
    }

    if( $format ) {

        $file = "neocaptar_".(LusiTime::now()->sec).$export_tools[$format]['extension'];
        $url  = "../neocaptar_documents/".$file;
        $path = "/tmp/".$file;

        $export_tools[$format]['convertor']($cables, $path, $user);

        $neocaptar->commit();
        $authdb->commit();
        $regdb->commit();

        NeoCaptarUtils::report_success(array('url' => $url, 'name' => $file));

    } else {

        $cables2return = array();
        $proj_id2title = array();

        foreach( $cables as $cable ) {
            $project = $cable->project();
            array_push($cables2return, NeoCaptarUtils::cable2array($cable,$project->id()));
            $proj_id2title[$project->id()] = $project->title();
        }

        $neocaptar->commit();
        $authdb->commit();
        $regdb->commit();

        NeoCaptarUtils::report_success( array(
            'cable' => $cables2return,
            'proj_id2title' => $proj_id2title ));
    }

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( RegDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }

?>
