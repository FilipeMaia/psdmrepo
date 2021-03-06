<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use LogBook\LogBookAuth;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use RegDB\RegDB;
use RegDB\RegDBException;

$instr = null;
if (isset($_GET['instr'])) {
    $instr = trim( $_GET['instr'] );
    if ($instr == '')
        die( "instrument name can't be empty" );
}
$is_location = isset( $_GET['is_location'] );

function experiment2json( $experiment ) {

    $instrument = $experiment->instrument();
    /*
    $experiment_url =
        "<a href=\"javascript:select_experiment(".
        $instrument->id().",'".$instrument->name()."',".
        $experiment->id().",'".$experiment->name()."')\" class=\"lb_link\">".
        $experiment->name().
        '</a>';
    */
    $experiment_url = "<a href=\"index.php?action=select_experiment_by_id&id={$experiment->id()}\" class=\"lb_link\">{$experiment->name()}</a>";
    $status = $experiment->in_interval( LusiTime::now());
    if ($status > 0) {
        $experiment_status = '<b><em style="color:gray">completed</em></b>';
    } elseif ($status < 0) {
        $experiment_status = '<b><em style="color:green">in preparation</em></b>';
    } else {
        $experiment_status = '<b><em style="color:red">on-going</em></b>';
    }
    if ($experiment->is_facility())
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
 * Return JSON objects with a list of experiments.
 */
try {
    RegDB::instance()->begin();

    if( is_null( $instr )) $all_experiments = RegDB::instance()->experiments();
    else                   $all_experiments = RegDB::instance()->experiments_for_instrument( $instr );

    // Leave only those experiments the logged user is authorizated to see
    //
    $experiments = array();
    foreach ($all_experiments as $e) {
        if (LogBookAuth::instance()->canRead( $e->id())) {
            if ($is_location xor $e->is_facility()) continue;
            array_push( $experiments, $e );
        }
    }

    // Proceed to the operation
    //
    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    $first = true;
    foreach ($experiments as $e) {
        if ($first) {
            $first = false;
            echo "\n".experiment2json( $e );
        } else {
            echo ",\n".experiment2json( $e );
        }
    }
    print <<< HERE
 ] } }
HERE;

    RegDB::instance()->commit();

} catch (LogBookException $e) { print $e->toHtml(); }
  catch (RegDBException   $e) { print $e->toHtml(); }

?>
