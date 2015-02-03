<?php

/**
 * This service will located and synchronzie orphan cables return a list of cable numbers which aren't associated
 * with the cable number allocation ranges.
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

    $prefixes = array();
    foreach( $neocaptar->cablenumber_prefixes() as $p )
        $prefixes[$p->name()] = $p;

    $uid = AuthDB::instance()->authName();
    foreach( $neocaptar->find_orphant_cables() as $prefix_name => $cables )
        foreach( $cables['in_range'] as $cable )
            $prefixes[$prefix_name]->synchronize_cable($cable,$uid);

    $prefix2array = NeoCaptarUtils::cablenumber_orphant2array($neocaptar);

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success(array('prefix' => $prefix2array));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
