<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'neocaptar/neocaptar.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

require_once( 'pdf-php/class.ezpdf.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use NeoCaptar\NeoCaptar;
use NeoCaptar\NeoCaptarUtils;
use NeoCaptar\NeoCaptarException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

/**
 * This service will produce a printable PDF document with cable labels
 * for both end of the cable.
 */
function report_error($msg) {
    print $msg;
    exit;
}
function label($pdf, $cable, $now, $src2dst=true) {

    $font_size = 6;
    $angle = 0;
    $max_length = 23+7;
    $first = 10;
    $second = 20;
    // Visible areas:
    //
    //   Left label: x:10..110, y: 150..250

//    $xmin = 0;
//    $xmax = 255; // experimentaly detected
//    $ymin = 0;
//    $ymax = 35;  // experimentaly detected
//
//    $pdf->setLineStyle(1);
//
//    $pdf->rectangle(  0, 0,110,35);
//    $pdf->rectangle(  5, 5,100,25);
//
//    $pdf->rectangle(145, 0,110,35);
//    $pdf->rectangle(150, 5,100,25);

    //$pdf->selectFont( './fonts/Courier.afm' );
    $pdf->selectFont( './fonts/Helvetica-Bold.afm' );
    $pdf->addText  (     $first, 29, $font_size, $cable->cable(),                                    $angle);
    $pdf->selectFont( './fonts/Helvetica.afm' );
    $pdf->addText  (    38, 29, $font_size, 'R00',                                                   $angle);
    //$pdf->addText  (    58, 29, $font_size, LusiTime::now()->toStringDay(),                        $angle);
    $pdf->addText  (    53, 29, $font_size, $cable->origin_pinlist(),                                $angle);
    $pdf->selectFont( './fonts/Helvetica-Bold.afm' );
    $pdf->addText  (145+ $first, 29, $font_size, $cable->cable(),                                    $angle);
    $pdf->selectFont( './fonts/Helvetica.afm' );
    $pdf->addText  (145+38, 29, $font_size, 'R00',                                                   $angle);
    //$pdf->addText  (145+58, 29, $font_size, LusiTime::now()->toStringDay(),                        $angle);
    $pdf->addText  (145+53, 29, $font_size, $cable->destination_pinlist(),                           $angle);

    $pdf->selectFont( './fonts/Helvetica-Bold.afm' );
    $pdf->addText  (     $first, 22, $font_size, "to:",                                              $angle);
    $pdf->selectFont( './fonts/Helvetica.afm' );
    $pdf->addText  (    $second, 22, $font_size, substr($cable->destination_name(), 0, $max_length), $angle);

    $pdf->selectFont( './fonts/Helvetica-Bold.afm' );
    $pdf->addText  (145+ $first, 22, $font_size, "to:",                                              $angle);
    $pdf->selectFont( './fonts/Helvetica.afm' );
    $pdf->addText  (145+$second, 22, $font_size, substr($cable->origin_name(), 0, $max_length),      $angle);

    $pdf->selectFont( './fonts/Helvetica-Bold.afm' );
    $pdf->addText  (     $first, 16, $font_size, "fr:",                                              $angle);
    $pdf->selectFont( './fonts/Helvetica.afm' );
    $pdf->addText  (    $second, 16, $font_size, substr($cable->origin_name(), 0, $max_length),      $angle);

    $pdf->selectFont( './fonts/Helvetica-Bold.afm' );
    $pdf->addText  (145+ $first, 16, $font_size, "fr:",                                              $angle);
    $pdf->selectFont( './fonts/Helvetica.afm' );
    $pdf->addText  (145+$second, 16, $font_size, substr($cable->destination_name(), 0, $max_length), $angle);

    $pdf->selectFont( './fonts/Helvetica-Bold.afm' );
    $pdf->addText  (     $first,  9, $font_size, "fn:",                                              $angle);
    $pdf->selectFont( './fonts/Helvetica.afm' );
    $pdf->addText  (    $second,  9, $font_size, substr($cable->func(), 0, $max_length),             $angle);

    $pdf->selectFont( './fonts/Helvetica-Bold.afm' );
    $pdf->addText  (145+ $first,  9, $font_size, "fn:",                                              $angle);
    $pdf->selectFont( './fonts/Helvetica.afm' );
    $pdf->addText  (145+$second,  9, $font_size, substr($cable->func(), 0, $max_length),             $angle);

    $pdf->addText  (     $first,  2, $font_size, substr($cable->device(), 0, $max_length),           $angle);
    $pdf->addText  (145+ $first,  2, $font_size, substr($cable->device(), 0, $max_length),           $angle);

//    $pdf->addText  (    25, 28, $font_size, $cable->cable(),            $angle);
//    $pdf->addText  (145+25, 28, $font_size, $cable->cable(),            $angle);
//
//    $pdf->addText  (     8, 20, $font_size, "to:", $angle);
//    $pdf->addText  (    25, 20, $font_size, $cable->destination_name(), $angle);
//
//    $pdf->addText  (145+ 8, 20, $font_size, "to:", $angle);
//    $pdf->addText  (145+25, 20, $font_size, $cable->origin_name(),      $angle);
//
//    $pdf->addText  (     8, 12, $font_size, "fr:",                      $angle);
//    $pdf->addText  (    25, 12, $font_size, $cable->origin_name(),      $angle);
//
//    $pdf->addText  (145+ 8, 12, $font_size, "fr:",                      $angle);
//    $pdf->addText  (145+25, 12, $font_size, $cable->destination_name(), $angle);
//
//    $pdf->addText  (    25,  4, $font_size,       $cable->device(),     $angle);
//    $pdf->addText  (145+25,  4, $font_size,       $cable->device(),     $angle);
}
try {

    $cable_id   = null;
    $project_id = null;
    if(isset($_GET['cable_id'])) {
        $cable_id = intval(trim($_GET['cable_id']));
        if(!$cable_id) report_error('empty or invalid value of parameter cable_id');
    } else if( isset($_GET['project_id'])) {
        $project_id = intval(trim($_GET['project_id']));
        if(!$project_id) report_error('empty or invalid value of parameter project_id');
    } else {
        report_error('please provide cable_id or project_id');
    }
	$authdb = AuthDB::instance();
	$authdb->begin();

    $neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

    $cables = array();
    if($cable_id) {
    	$cable = $neocaptar->find_cable_by_id($cable_id);
        if(is_null($cable)) report_error('no cable exists for id: '.$cable_id);
        array_push($cables, $cable);
    } else {
        $project = $neocaptar->find_project_by_id($project_id);
        if(is_null($project)) report_error('no project exists for id: '.$project_id);
        foreach($project->cables() as $cable) {
            if( $cable->status() != 'Planned') array_push($cables, $cable);
        }
    }

    // IMPORTANT: set printer driver parameters to:
    // width: 3.54 inches, height: 1.0 inch, all margins (top,bottom,left,right) are set to 0
    // orientation: auto portrait/landscape , size option: actual size
    //
    $pdf = new Cezpdf(array(3.54*2.54,0.50*2.54));
    $pdf->ezSetMargins(0,0,0,0);
    $pdf->setColor(0,0,0);

    $first = true;
    foreach($cables as $cable) {
        if($first) $first = false;
        else       $pdf->ezSetDY(-48);
        label($pdf,$cable,true);
    }

    $pdf->stream();

    $neocaptar->commit();
	$authdb->commit();

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error($e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error($e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error($e->toHtml()); }
  
?>
