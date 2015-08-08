<?php

/**
 * This service will remove the specified range from the prefix.
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

    $prefix   = NeoCaptarUtils::get_param_GET('prefix');
    $range_id = NeoCaptarUtils::get_param_GET('range_id');

    $authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

    $prefix_obj = $neocaptar->find_cablenumber_prefix_by_name($prefix);
    if(is_null($prefix_obj))
        NeoCaptarUtils::report_error("no such prefix found in the database");

    $prefix_obj->delete_range($range_id);

    $range2array = NeoCaptarUtils::cablenumber_ranges2array($prefix_obj);

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success(array('range' => $range2array));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
