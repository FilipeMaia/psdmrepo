<?php

/**
 * This service will return an access control lists.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use DataPortal\NeoCaptar;
use DataPortal\NeoCaptarUtils;
use DataPortal\NeoCaptarException;

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {
	$authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

    $access2array = array(
        'administrator' => array(),
        'projmanager' => array()
    );
    foreach( $neocaptar->users() as $u )
        array_push($access2array[$u->role()], NeoCaptarUtils::user2array($u));

	$authdb->commit();
	$neocaptar->commit();

    NeoCaptarUtils::report_success( array( 'access' => $access2array ));

} catch( AuthDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException   $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
