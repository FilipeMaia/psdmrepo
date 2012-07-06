<?php

/**
 * This service will update parameters of the specified cable numbers
 * allocation and return an updated object.
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

    $prefix2location = $_GET;

    $authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

    foreach( $prefix2location as $prefix => $location ) {
        $prefix   = strtoupper(trim($prefix));
        if(strlen($prefix) != 2)
            NeoCaptarUtils::report_error("illegal value for parameters: each prefix must have exactly 2 symbols");
        $prefix_obj = $neocaptar->find_cablenumber_prefix_by_name($prefix);
        if( is_null($prefix_obj)) $prefix_obj = $neocaptar->add_cablenumber_prefix($prefix);
        $location = trim($location);
        if( $location != '' ) $prefix_obj->add_location($location);
    }

    $prefix2array = NeoCaptarUtils::cablenumber_prefixes2array($neocaptar);

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success(array('prefix' => $prefix2array));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
