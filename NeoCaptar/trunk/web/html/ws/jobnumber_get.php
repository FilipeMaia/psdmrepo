<?php

/**
 * This service will return a list of job number allocation.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'neocaptar/neocaptar.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use NeoCaptar\NeoCaptar;
use NeoCaptar\NeoCaptarUtils;
use NeoCaptar\NeoCaptarException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {
	$authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$jobnumbers_as_array            = array();
    $jobnumber_allocations_as_array = array();

    foreach( $neocaptar->jobnumber_allocations() as $j ) {
        array_push($jobnumbers_as_array, NeoCaptarUtils::jobnumber2array($j));
        foreach( $j->jobnumbers() as $ja )
            array_push($jobnumber_allocations_as_array, NeoCaptarUtils::jobnumber_allocation2array($ja));
    }

    $authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success(
        array(
            'jobnumber'            => $jobnumbers_as_array,
            'jobnumber_allocation' => $jobnumber_allocations_as_array ));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
