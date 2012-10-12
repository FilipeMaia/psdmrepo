<?php

require_once( 'regdb/regdb.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use RegDB\RegDBHtml;

use FileMgr\FileMgrIfaceCtrlWs;
use FileMgr\FileMgrException;

use LusiTime\LusiTime;

/* ----------------------------------------
 * Parse mandatory parameters of the script
 * ----------------------------------------
 */
if( !isset( $_GET['instr'] )) die( "no valid instrument name" );
$instrument = trim( $_GET['instr'] );

if( !isset( $_GET['exp'] )) die( "no valid experiment name" );
$experiment = trim( $_GET['exp'] );

/* --------------------------------
 * Parse optional filter parameters
 * --------------------------------
 */
if( isset( $_GET['begin_run'] )) {
    $begin_run = 0;
    if( 1 != sscanf( trim( $_GET['begin_run'] ), "%u", $begin_run )) die( "begin run must be a number" );
}
if( isset( $_GET['end_run'] )) {
    $end_run = 0;
    if( 1 != sscanf( trim( $_GET['end_run'] ), "%u", $end_run )) die( "end run must be a number" );
    if( isset( $begin_run ) && ( $end_run < $begin_run )) die( "end run must be equal or greater than the begin run" );
}

if( isset( $_GET['status'] )) {
    $status = trim( $_GET['status'] );
    if( $status == '' ) die( "status can not be an empty string" );
}

function parse_time( $parname ) {
    $result = LusiTime::parse( trim( $_GET[$parname] ));
    if( is_null( $result )) die( "invalid format of the {$parname} parameter" );
    return $result;
}

if( isset( $_GET['begin_created'] )) {
    $begin_created = parse_time( 'begin_created' );
}
if( isset( $_GET['end_created'] )) {
    $end_created = parse_time( 'end_created' );
    if( isset( $begin_created ) && $end_created->less( $begin_created ) ) die( "end_created must be equal or greater than the begin_created" );
}

if( isset( $_GET['begin_started'] )) {
    $begin_started = parse_time( 'begin_started' );
}
if( isset( $_GET['end_started'] )) {
    $end_started = parse_time( 'end_started' );
    if( isset( $begin_started ) && $end_started->less( $begin_started ) ) die( "end_started must be equal or greater than the begin_started" );
}

if( isset( $_GET['begin_stopped'] )) {
    $begin_stopped = parse_time( 'begin_stopped' );
}
if( isset( $_GET['end_stopped'] )) {
    $end_stopped = parse_time( 'end_stopped' );
    if( isset( $begin_stopped ) && $end_stopped->less( $begin_stopped ) ) die( "end_stopped must be equal or greater than the begin_stopped" );
}

/*
 * Generate preformatted output with a possibility to limit the length
 * of the output string. The later option will be turned on if overriding
 * the default value of the 'width' parameter of the function.
 */
function bold( $str ) {
    return '<b>'.$str.'</b>';
}

function pre( $str, $width=null ) {
    if( is_null( $width )) return '<pre>'.$str.'</pre>';
    return '<pre>'.sprintf( "%{$width}s", $str ).'</pre>';
}

/* -----------------------------------------
 * Color generator for various status values
 * -----------------------------------------
 */
function color_for( $status ) {
    if( $status == 'Translation_Error' ) return '#ff0000';  // pure red
    if( $status == 'H5Dir_Error'       ) return '#ff0000';  // pure red
    if( $status == 'Archive_Error'     ) return '#ff0000';  // pure red
    if( $status == 'Being_Translated'  ) return '#336600';  // deep green
    if( $status == 'Complete'          ) return '#a0a0a0';  // deep grey
    return null;
}

/* ------------------------------------------------------------------------
 * Optional filtering of requests. The filter is turned on if any filtering
 * parameters are passed to the script.
 * ------------------------------------------------------------------------
 */
function apply_filter( $in ) {

    global $begin_run,     $end_run,
           $begin_created, $end_created,
           $begin_started, $end_started,
           $begin_stopped, $end_stopped,
           $status;

    $filter_is_on =
        isset( $begin_run     ) || isset( $end_run     ) ||
        isset( $begin_created ) || isset( $end_created ) ||
        isset( $begin_started ) || isset( $end_started ) ||
        isset( $begin_stopped ) || isset( $end_stopped ) ||
        isset( $status );

    if( !$filter_is_on ) return $in;

    $out = array();
    foreach( $in as $i ) {
        if( isset( $begin_run ) && ( (int)($i->run) <  $begin_run  )) continue;
        if( isset( $end_run   ) && ( (int)($i->run) >  $end_run  )) continue;

        if( isset( $status ) && ( $i->status != $status )) continue;

        if( isset( $i->created )) {
            if( isset( $begin_created ) && LusiTime::parse( $i->created )->less          ( $begin_created )) continue;
            if( isset( $end_created   ) && LusiTime::parse( $i->created )->greaterOrEqual( $end_created   )) continue;
        }
        if( isset( $i->started )) {
            if( isset( $begin_started ) && LusiTime::parse( $i->started )->less          ( $begin_started )) continue;
            if( isset( $end_started   ) && LusiTime::parse( $i->started )->greaterOrEqual( $end_started   )) continue;
        }
        if( isset( $i->stopped )) {
            if( isset( $begin_stopped ) && LusiTime::parse( $i->stopped )->less          ( $begin_stopped )) continue;
            if( isset( $end_stopped   ) && LusiTime::parse( $i->stopped )->greaterOrEqual( $end_stopped   )) continue;
        }
        array_push( $out, $i );
    }
    return $out;
}

/* --------------------------------------------------------------------
 * Optional sorting of requests. The filter is turned on if any sorting
 * parameters are passed to the script.
 * 
 * NOTE: The default sort order is to sort by runs and then sort
 *        within each run by requests creation time.
 * --------------------------------------------------------------------
 */
function apply_sort( $in ) {
    $run2created2request = array();
    foreach( $in as $i ) {
        $run2created2request[$i->run][$i->created] = $i;
    }
    $out = array();
    $runs = array_keys( $run2created2request );
    sort( $runs );
    foreach( $runs as $r ) {
        $created = array_keys( $run2created2request[$r] );
        sort( $created );
        foreach( $created as $c ) {
            array_push( $out, $run2created2request[$r][$c] );
        }
    }
    return $out;
}

/* -----------------------------
 * Begin the main algorithm here
 * -----------------------------
 */
try {

    $requests = apply_sort(
        apply_filter (
            FileMgrIfaceCtrlWs::experiment_requests (
                $instrument,
                $experiment
               )
        )
    );

    $num_rows = 20 * count( $requests );

    $con = new RegDBHtml( 0, 0, 900, $num_rows );

    $row = 0;
    foreach( $requests as $r ) {

        $log_url = '';
        if( $r->log_url != '' )
            $log_url = pre( '<a href="'.$r->log_url.'" target="_blank" title="click to open the log file in a separate tab or window">View</a>' );

        $color = color_for( $r->status );

        $con->value(   5, $row, pre( $r->id ), $color);
        $con->value(  65, $row, pre( $r->run ), $color);
        $con->value( 125, $row, bold( pre( $r->status )), $color );
        $con->value( 280, $row, pre( $r->priority ), $color);
        $con->value( 340, $row, pre( $r->created ), $color);
        $con->value( 510, $row, pre( $r->started ), $color);
        $con->value( 675, $row, pre( $r->stopped ), $color);
        $con->value( 850, $row, $log_url );

        $row += 20;
    }
    print $con->html();

} catch( FileMgrException $e ) {
    echo $e->toHtml();
}
?>