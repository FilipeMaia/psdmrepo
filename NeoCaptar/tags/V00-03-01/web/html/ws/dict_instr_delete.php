<?php

/**
 * This service will delete the specified item instr from the dictionary
 * and return an empty dictionary.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'neocaptar/neocaptar.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use NeoCaptar\NeoCaptar;
use NeoCaptar\NeoCaptarUtils;
use NeoCaptar\NeoCaptarException;

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {

    $scope = NeoCaptarUtils::get_param_GET('scope');
    $id    = NeoCaptarUtils::get_param_GET('id');

    if( $scope != 'instr' )
        NeoCaptarUtils::report_error(($scope==''?'empty':'illegal').' value of the scope parameter found in the request');

	$authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

	$neocaptar->delete_dict_instr_by_id( $id );

	$instrs = NeoCaptarUtils::dict_instrs2array($neocaptar);

    $neocaptar->commit();

	$authdb->commit();

    NeoCaptarUtils::report_success( array( 'instr' => $instrs ));

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
