<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

require_once( 'pdf-php/class.ezpdf.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use DataPortal\NeoCaptar;
use DataPortal\NeoCaptarUtils;
use DataPortal\NeoCaptarException;

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
    /*
    $pdf->selectFont( './fonts/Courier-Bold.afm' );
    $pdf->ezSetDY(30); $pdf->ezText($cable->cable(),                               9, array( 'left' => 48 ));
    $pdf->selectFont( './fonts/Courier.afm' );
    $pdf->ezSetDY(-2); $pdf->ezText($cable->device().' R0 '.$now->toStringDay_1(), 8, array( 'left' =>  0 ));
    if($src2dst ) {
    $pdf->ezSetDY(-2); $pdf->ezText('to: '.$cable->destination_name(),             8, array( 'left' =>  0 ));
    $pdf->ezSetDY(-2); $pdf->ezText('fr: '.$cable->origin_name(),                  8, array( 'left' =>  0 ));
    } else {
    $pdf->ezSetDY(-2); $pdf->ezText('to: '.$cable->origin_name(),                  8, array( 'left' =>  0 ));
    $pdf->ezSetDY(-2); $pdf->ezText('fr: '.$cable->destination_name(),             8, array( 'left' =>  0 ));
    }
    $pdf->ezSetDY(-2); $pdf->ezText('fn: '.$cable->func(),                         8, array( 'left' =>  0 ));
    $pdf->ezSetDY(-2); $pdf->ezText('JOB#: ',                                      8, array( 'left' => 48 ));
    $pdf->selectFont( './fonts/Courier-Bold.afm' );
    $pdf->ezSetDY( 9); $pdf->ezText($cable->job(),                                 9, array( 'left' => 74 ));
     */
    $device_dpi = 300.0;
    $device_dots_cm = $device_dpi / 2.54;
    $dot_size_mm = 1;
    $angle = 0;
    $font_size = 8;
    $pdf->selectFont( './fonts/Courier-Bold.afm' );

    // Visible areas:
    //
    //  Left label: x:10..110, y: 150..250
    $xmin = 0;
    $xmax = 255; // experimentaly detected
    $ymin = 0;
    $ymax = 35;  // experimentaly detected
    $pdf->setLineStyle(1);

//    $pdf->rectangle(  0, 0,110,35);
//    $pdf->rectangle(  5, 5,100,25);
//
//    $pdf->rectangle(145, 0,110,35);
//    $pdf->rectangle(150, 5,100,25);

    $pdf->addText  (    25, 28, $font_size, $cable->cable(),            $angle);
    $pdf->addText  (145+25, 28, $font_size, $cable->cable(),            $angle);

    $pdf->addText  (     8, 20, $font_size, "to:", $angle);
    $pdf->addText  (    25, 20, $font_size, $cable->destination_name(), $angle);

    $pdf->addText  (145+ 8, 20, $font_size, "to:", $angle);
    $pdf->addText  (145+25, 20, $font_size, $cable->destination_name(), $angle);

    $pdf->addText  (     8, 12, $font_size, "fr:",                      $angle);
    $pdf->addText  (    25, 12, $font_size, $cable->origin_name(),      $angle);

    $pdf->addText  (145+ 8, 12, $font_size, "fr:",                      $angle);
    $pdf->addText  (145+25, 12, $font_size, $cable->origin_name(),      $angle);

    $pdf->addText  (    25,  4, $font_size,       $cable->device(),             $angle);
    $pdf->addText  (145+25,  4, $font_size,       $cable->device(),             $angle);
    /*
    for( $y = $ymin; $y <= $ymax; $y += 5 ) $pdf->line($xmin,$y,$xmax,$y);
    for( $x = $xmin; $x <= $xmax; $x += 5 ) $pdf->line($x,$ymin,$x,$ymax);
    $pdf->addText(10, 10,$font_size,"10",$angle);
    $pdf->addText(102,10,$font_size,"110",$angle);
    $pdf->addText( 52,20,$font_size,"w:{$ymax}",$angle);
    $pdf->addText(150,10,$font_size,"h:{$xmax}",$angle);
    $pdf->addText(200,20,$font_size,"220",$angle);
    */
}
try {

    if(!isset($_GET['cable_id'])) report_error('missing parameter cable_id');
    $cable_id = intval(trim($_GET['cable_id']));
    if(!$cable_id) report_error('empty or invalid value of parameter cable_id');

	$authdb = AuthDB::instance();
	$authdb->begin();

    $neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$cable = $neocaptar->find_cable_by_id($cable_id);
    if(is_null($cable)) report_error('no cable exists for id: '.$cable_id);

    $now = LusiTime::now();

    // IMPORTANT: set printer driver parameters to:
    // width: 3.54 inches, height: 1.0 inch, all margins (top,bottom,left,right) are set to 0
    // orientation: auto portrait/landscape , size option: actual size
    //
    $pdf = new Cezpdf(array(3.54*2.54,0.50*2.54));
    $pdf->ezSetMargins(0,0,0,0);
    $pdf->setColor(0,0,0);

    //$pdf->ezSetDY(-70);
    label($pdf,$cable,$now,true);
    $pdf->ezSetDY(-48);
    label($pdf,$cable,$now,false);


    $pdf->stream();

    $neocaptar->commit();
	$authdb->commit();

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error($e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error($e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error($e->toHtml()); }
  
?>
