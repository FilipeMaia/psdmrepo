<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use LogBook\LogBookAuth;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use RegDB\RegDB;
use RegDB\RegDBException;

$instr = null;
if( isset( $_GET['instr'] )) {
    $instr = trim( $_GET['instr'] );
    if( $instr == '' ) {
        die( "instrument name can't be empty" );
    }
}

function experiment2json( $experiment ) {

    $instrument = $experiment->instrument();
    $experiment_url = "<a href=\"index.php?action=select_experiment_by_id&id={$experiment->id()}\" class=\"lb_link\">{$experiment->name()}</a>";
    $status = $experiment->in_interval( LusiTime::now());
    if( $status > 0 ) {
        $experiment_status = '<b><em style="color:gray">completed</em></b>';
    } else if( $status < 0 ) {
        $experiment_status = '<b><em style="color:green">in preparation</em></b>';
    } else {
        $experiment_status = '<b><em style="color:red">on-going</em></b>';
    }
    if( $experiment->is_facility())
        $obj = array (
            "location"           => $experiment->instrument()->name(),
            "facility"           => $experiment_url,
            "name"               => $experiment->name(),
            "id"                 => $experiment->id(),
            "registration_time"  => $experiment->registration_time()->toStringShort(),
            "description"        => substr( $experiment->description(), 0, 72 )."..."
        );
    else
        $obj = array (
            "instrument"  => $experiment->instrument()->name(),
            "experiment"  => $experiment_url,
            "name"        => $experiment->name(),
            "id"          => $experiment->id(),
            "status"      => $experiment_status,
            "begin_time"  => $experiment->begin_time()->toStringShort(),
            "end_time"    => $experiment->end_time()->toStringShort(),
            "registration_time"   => $experiment->registration_time()->toStringShort(),
            "description" => substr( $experiment->description(), 0, 72 )."..."
        );
    return json_encode( $obj );
}

/*
 * Return JSON objects with the name of the current experiment.
 */
try {
    RegDB::instance()->begin();

    $last_switch = RegDB::instance()->last_experiment_switch( $instr ) or
        die( "no current experiment for instrument: {$instr}" );

    $last_experiment = RegDB::instance()->find_experiment_by_id( $last_switch['exper_id'] ) or
        die( "failed to find experiment for id=".$last_switch['exper_id'] );

    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    echo '{ "ResultSet": { "Result": '.experiment2json( $last_experiment ).' } }';

    RegDB::instance()->commit();

} catch (LogBookException $e) { print $e->toHtml(); }
  catch (RegDBException   $e) { print $e->toHtml(); }

?>
