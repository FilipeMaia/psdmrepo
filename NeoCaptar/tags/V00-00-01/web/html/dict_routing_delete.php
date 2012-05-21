<?php

/**
 * This service will delete the specified item routing from the dictionary
 * and return an empty dictionary.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'neocaptar/neocaptar.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use NeoCaptar\NeoCaptar;
use NeoCaptar\NeoCaptarUtils;
use NeoCaptar\NeoCaptarException;

use LusiTime\LusiTimeException;

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {

    $id    = NeoCaptarUtils::get_param_GET('id');
    $scope = NeoCaptarUtils::get_param_GET('scope');
    if( $scope != 'routing' )
        NeoCaptarUtils::report_error(($scope==''?'empty':'illegal').' value of the scope parameter found in the request');

    $authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();
	$neocaptar->delete_dict_routing_by_id( $id );
	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success();

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
