<?php

require_once( 'logbook/logbook.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookAuth;
use LogBook\LogBookException;

use RegDB\RegDBHtml;
use RegDB\RegDBException;

/*
 * This script will process a request for displaying a status of an experimental
 * facility.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "experiment/facility identifier can't be empty" );
} else
    die( "no valid experiment/facility identifier" );


/* Proceed with the operation
 */
try {
    LogBook::instance()->begin();

    $experiment = LogBook::instance()->find_experiment_by_id( $id )  or die( "no such experiment" );
    if( !$experiment->is_facility())
        die( "the identifier {$id} corresponds to an experiment not to a facility" ); 

    $instrument = $experiment->instrument();

    // Check for the authorization
    //
    if( !LogBookAuth::instance()->canRead( $experiment->id())) {
        print( LogBookAuth::reporErrorHtml(
            'You are not authorized to access any information about the experiment' ));
        exit;
    }

    // Proceed to the operation
    //
    $shift = $experiment->find_last_shift();

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past


    $all_shifts_url = "<a href=\"javascript:list_shifts()\" class=\"lb_link\">See List of all shifts</a>";
    $shift_url = 'See last shift';
    if( !is_null( $shift )) {
        $shift_url = "<a href=\"javascript:select_shift({$shift->id()})\" class=\"lb_link\">See last shift</a>";
    }

    $con = new RegDBHtml( 0, 0, 900, 250 );
    echo $con
        ->label       ( 250,   0, 'Description')
        ->container_1 ( 250,  20, '<pre style="padding:4px; font-size:14px;">'.$experiment->description().'</pre>', 500, 140, false, '#efefef' )

        ->label       (   0,   0, 'Leader:'     )->value( 150,   0, $experiment->leader_account())
        ->label       (   0,  20, 'POSIX Group:')->value( 150,  20, $experiment->POSIX_gid())

        ->label       (   0,  60, $shift_url, false )
        ->label       (   0,  80, $all_shifts_url, false )

        ->label       ( 250, 180, 'Contact Info')
        ->container_1 ( 250, 200, '<pre style="padding:4px; font-size:14px;">'.$experiment->contact_info().'</pre>', 500, 50, false, '#efefef' )

        ->html();

    LogBook::instance()->commit();

} catch( LogBookException $e ) { print $e->toHtml(); }
  catch( RegDBException   $e ) { print $e->toHtml(); }

?>