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

    $pdf = new Cezpdf(/*'letter', 'portrait'*/);
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
