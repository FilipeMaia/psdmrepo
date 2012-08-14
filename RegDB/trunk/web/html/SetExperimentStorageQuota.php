<?php

require_once( 'regdb/regdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;


/*
 * This script will process a request for setting/updating a value
 * of the 'MEDIUM-TERM-DISK-QUOTA' parameter for an experiment.
 */
/*
 * NOTE: Can not return JSON MIME type because of the following issue:
 *       http://jquery.malsup.com/form/#file-upload
 *
header( "Content-type: application/json" );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
*/

/* Package the error message into a JSON object and return the one
 * back to a caller. The script's execution will end at this point.
 */
function report_error( $msg ) {
    $status_encoded = json_encode( "error" );
    $msg_encoded = json_encode( $msg );
    print <<< HERE
<textarea>
{
  "Status": {$status_encoded},
  "Message": {$msg_encoded}
}
</textarea>
HERE;
    exit;
}

if( isset( $_POST['exper_id'] )) {
    $exper_id = trim( $_POST['exper_id'] );
    if( $exper_id == '' ) report_error( "experiment identifier can't be empty" );
} else report_error( "no valid experiment identifier" );

$short_ctime = null;
if( isset( $_POST['short_ctime'] )) {
    $str = trim( $_POST['short_ctime'] );
    if( $str ) {
        $short_ctime = LusiTime::parse($str);
        if( is_null($short_ctime )) report_error( "invalid value for the SHORT-TERM CTIME overrride parameter" );
    }
}
$short_retention = null;
if( isset( $_POST['short_retention'] )) {
    $short_retention = intval( trim( $_POST['short_retention'] ));
}
$medium_quota = null;
if( isset( $_POST['medium_quota'] )) {
    $medium_quota = intval( trim( $_POST['medium_quota'] ));
}
$medium_ctime = null;
if( isset( $_POST['medium_ctime'] )) {
    $str = trim( $_POST['medium_ctime'] );
    if( $str ) {
        $medium_ctime = LusiTime::parse($str);
        if( is_null($medium_ctime )) report_error( "invalid value for the MEDIUM-TERM CTIME overrride parameter" );
    }
}
$medium_retention = null;
if( isset( $_POST['medium_retention'] )) {
    $medium_retention = intval( trim( $_POST['medium_retention'] ));
}

try {
	$regdb = RegDB::instance();
	$regdb->begin();

	if( !RegDBAuth::instance()->canEdit())
		report_error( 'not autorized to modify MEDIUM-TERM storage quota parameter for the experiment' );

	$experiment = $regdb->find_experiment_by_id( $exper_id );
	if( !$experiment ) report_error( 'no such experiment' );

    function set_param($name, $value) {
        global $experiment;
        if(is_null($value)) return;
     	$param = $experiment->set_param( $name, $value );
        if( is_null( $param )) report_error( "failed to set value='{$value}' to experiment-specific parameter {$name}" );
    }
    set_param('SHORT-TERM-DISK-QUOTA-CTIME',      $short_ctime ? $short_ctime->toStringDay() : null);
    set_param('SHORT-TERM-DISK-QUOTA-RETENTION',  $short_retention);
    set_param('MEDIUM-TERM-DISK-QUOTA',           $medium_quota);
    set_param('MEDIUM-TERM-DISK-QUOTA-CTIME',     $medium_ctime ? $medium_ctime->toStringDay() : null );
    set_param('MEDIUM-TERM-DISK-QUOTA-RETENTION', $medium_retention);

	$regdb->commit();

    $status_encoded = json_encode( "success" );
    print <<< HERE
<textarea>
{
  "Status": {$status_encoded}
}
</textarea>
HERE;

} catch( RegDBException    $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }
  catch( LusiTimeException $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>