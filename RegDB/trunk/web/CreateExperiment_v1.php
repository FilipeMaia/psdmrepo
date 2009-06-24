<?php

require_once('RegDB.inc.php');
require_once('RegDBHtml.php');

/*
 * This script will lay out a form for creating a new experiment.
 */

/* Proceed with the operation
 */
try {
    $regdb = new RegDB();
    $regdb->begin();
    $instrument_names = $regdb->instrument_names();
    $posix_groups = $regdb->posix_groups();

    $now = new DateTime();
    $now_str = $now->format(DateTime::ISO8601);
    $now_str[10] = ' ';  // get rid of date-time separator 'T'
    $logged_user = $_SERVER['WEBAUTH_USER'];

/*
    $experiment = $regdb->find_experiment_by_id( $id )
        or die( "no such experiment" );

    $instrument = $experiment->instrument();
    $group      = $experiment->POSIX_gid();

    $instrument_url =
        "<a href=\"javascript:view_instrument(".$instrument->id().",'".$instrument->name()."')\">".$instrument->name().'</a>';

    $group_url =
        "<a href=\"javascript:view_group('".$group."')\">".$group.'</a>';
*/
    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    /* Create a container with standard fields
     */
    $con = new RegDBHtml( 0, 0, 700, 250 );
    echo $con
        ->label         ( 300,   0, 'Description')
        ->label         (   0,  25, 'Experiment: ' )
        ->value_input   ( 100,  25, 'experiment_name' )
        ->textarea_input( 300,  25, 'description', 500, 125 )
        ->label         (   0,  50, 'Instrument: ' )
        ->select_input  ( 100,  50, 'instrument_name', $instrument_names )
        ->label         (   0, 100, 'Begin Time: ' )
        ->value_input   ( 100, 100, 'begin_time', $now_str )
        ->label         (   0, 125, 'End Time: '   )
        ->value_input   ( 100, 125, 'end_time', $now_str )
        ->label         ( 300, 175, 'Contact Info' )
        ->label         (   0, 200, 'POSIX Group: ' )
        ->select_input  ( 100, 200, 'group', $posix_groups )
        ->textarea_input( 300, 200, 'contact', 500, 50 )
        ->label         (   0, 225, 'Leader: '     )
        ->value_input   ( 100, 225, 'leader', $logged_user )
        ->html();

    $regdb->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>