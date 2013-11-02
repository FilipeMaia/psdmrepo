<?php

/**
 * This service will set/toggle the specified privilege of a user account
 * and return an updated list.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'neocaptar/neocaptar.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use NeoCaptar\NeoCaptar;
use NeoCaptar\NeoCaptarUtils;
use NeoCaptar\NeoCaptarException;


header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

try {
    $uid  = NeoCaptarUtils::get_param_GET('uid');
    $name = NeoCaptarUtils::get_param_GET('name');

    $authdb = AuthDB::instance();
    $authdb->begin();

    $neocaptar = NeoCaptar::instance();
    $neocaptar->begin();

    if( !$neocaptar->is_administrator())
        NeoCaptarUtils::report_error("your account not authorized for the operation");

    $user = $neocaptar->find_user_by_uid( $uid );
    if( is_null($user)) NeoCaptarUtils::report_error("no such user: {$uid}");

    switch($name) {
        case 'dict_priv':
            $user->set_dict_priv( !$user->has_dict_priv());
            break;
        default:
            NeoCaptarUtils::report_error("unknown privilege requested: {$name}");
            break;
    }
    $access2array = NeoCaptarUtils::access2array($neocaptar->users());

    $authdb->commit();
    $neocaptar->commit();

    NeoCaptarUtils::report_success( array( 'access' => $access2array ));

} catch( AuthDBException    $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  catch( NeoCaptarException $e ) { NeoCaptarUtils::report_error( $e->toHtml()); }
  
?>
