<?php

/**
 * This service will create a new access controll entry and return an updated
 * access control list.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;


use DataPortal\NeoCaptar;
use DataPortal\NeoCaptarUtils;
use DataPortal\NeoCaptarException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use RegDB\RegDB;
use RegDB\RegDBException;

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {
    $uid  = NeoCaptarUtils::get_param_GET('uid');
    $role = NeoCaptarUtils::get_param_GET('role');

    $authdb = AuthDB::instance();
	$authdb->begin();

	$neocaptar = NeoCaptar::instance();
	$neocaptar->begin();

    if( !$neocaptar->is_administrator())
        NeoCaptarUtils::report_error("your account not authorized for the operation");

    $regdb = RegDB::instance();
    $regdb->begin();

    $user = $regdb->find_user_account( $uid );
    if( is_null($user)) NeoCaptarUtils::report_error("no such user: {$uid}");

    $user = $neocaptar->add_user($user['uid'], $user['gecos'], $role);

    $access2array = array(
        'administrator' => array(),
        'projmanager' => array()
    );
    foreach( $neocaptar->users() as $u )
        array_push($access2array[$u->role()], NeoCaptarUtils::user2array($u));

	$authdb->commit();
	$neocaptar->commit();
    $regdb->commit();

    NeoCaptarUtils::report_success( array( 'access' => $access2array ));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( RegDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
