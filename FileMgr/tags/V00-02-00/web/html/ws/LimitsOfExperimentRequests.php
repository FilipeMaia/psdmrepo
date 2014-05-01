<?php
    
require_once( 'filemgr/filemgr.inc.php' );

use FileMgr\FileMgrIfaceCtrlWs;
use FileMgr\FileMgrException;

/* ----------------------------------------
 * Parse mandatory parameters of the script
 * ----------------------------------------
 */
if( !isset( $_GET['instr'] )) die( "no valid instrument name" );
$instrument = trim( $_GET['instr'] );

if( !isset( $_GET['exp'] )) die( "no valid experiment name" );
$experiment = trim( $_GET['exp'] );

/* -----------------------------
 * Begin the main algorithm here
 * -----------------------------
 */
try {

	$begin_run     = PHP_INT_MAX;
	$end_run       = 0;
	$begin_created = '9999-99-99 99:99:99';
	$end_created   = '0000-00-00 00:00:00';
	$begin_started = '9999-99-99 99:99:99';
	$end_started   = '0000-00-00 00:00:00';
	$begin_stopped = '9999-99-99 99:99:99';
	$end_stopped   = '0000-00-00 00:00:00';

	function update_limits( &$begin, &$end, $v ) {
		if( !$v ) return;
        if( $v < $begin ) $begin = $v;
        if( $v > $end   ) $end   = $v;
	}

	// Sort through requests and find a range for each property
	//
    $requests = FileMgrIfaceCtrlWs::experiment_requests ( $instrument, $experiment );
    foreach( $requests as $r ) {
        update_limits( $begin_run,     $end_run,    (int)$r->run );
        update_limits( $begin_created, $end_created,     $r->created );
        update_limits( $begin_started, $end_started,     $r->started );
        update_limits( $begin_stopped, $end_stopped,     $r->stopped );
    }
    if(( $begin_started == '9999-99-99 99:99:99' ) ||
       ( $begin_stopped == '9999-99-99 99:99:99' )) {
    	$begin_started = '';
    	$end_started   = '';
    	$begin_stopped = '';
    	$end_stopped   = '';
       }

    // Proceed to the operation
    //
    header( "Content-type: application/json" );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;

    echo json_encode(
        array (
            "begin_run"     => $begin_run,
            "end_run"       => $end_run,
            "begin_created" => $begin_created,
            "end_created"   => $end_created,
            "begin_started" => $begin_started,
            "end_started"   => $end_started,
            "begin_stopped" => $begin_stopped,
            "end_stopped"   => $end_stopped
        )
    );

    print <<< HERE
 ] } }
HERE;

} catch( FileMgrException $e ) {
	echo $e->toHtml();
}
?>