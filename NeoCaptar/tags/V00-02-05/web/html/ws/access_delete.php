<?php

/**
 * This service will delete an existing account from the control list and return
 * an updated list.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'neocaptar/neocaptar.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use NeoCaptar\NeoCaptar;
use NeoCaptar\NeoCaptarUtils;
use NeoCaptar\NeoCaptarException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use RegDB\RegDB;
use RegDB\RegDBException;

header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {
    $uid  = NeoCaptarUtils::get_param_GET('uid');

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

    $projects = $neocaptar->projects(null,$user['uid']);
    $num_projects = count($projects);
    if($num_projects) NeoCaptarUtils::report_error("the account can't be deleted because its associated with {$num_projects} project(s)");

    $neocaptar->delete_user($user['uid']);

    $access2array = NeoCaptarUtils::access2array($neocaptar->users());

    $authdb->commit();
    $neocaptar->commit();
    $regdb->commit();

    NeoCaptarUtils::report_success( array( 'access' => $access2array ));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( LusiTimeException  $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( RegDBException     $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
