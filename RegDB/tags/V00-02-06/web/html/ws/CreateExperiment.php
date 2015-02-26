<?php

require_once( 'lusitime/lusitime.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBHtml;
use RegDB\RegDBException;

/*
 * This script will lay out a form for creating a new experiment.
 */

if( !RegDBAuth::instance()->canEdit()) {
    print( RegDBAuth::reporErrorHtml(
        'You are not authorized to manage the contents of the Experiment Registry Database'));
    exit;
}

/* Proceed with the operation
 */
try {

    RegDB::instance()->begin();

    $instrument_names = RegDB::instance()->instrument_names();
    $posix_groups     = RegDB::instance()->posix_groups();

    // Get the currnt time in the ISO format, then stripe out
    // the date-time separator 'T' and timezone.
    //
    $now = LusiTime::now();
    $now_str = $now->toStringShort();

    $logged_user = $_SERVER['WEBAUTH_USER'];

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
    
    /* Create a container with standard fields
     */
    $con = new RegDBHtml( 0, 0, 800, 250 );
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

    RegDB::instance()->commit();

} catch (LusiTimeException $e) { print $e->toHtml(); }
  catch (RegDBException    $e) { print $e->toHtml(); }

?>
