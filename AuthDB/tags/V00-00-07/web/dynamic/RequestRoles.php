<?php

require_once('AuthDB/AuthDB.inc.php');

/*
 * This script will process requests for various information stored in the database.
 * The result will be returned as JSON object (array).
 */
if( !isset( $_GET['type'] )) die( "no valid information type in the request" );
$type = trim( $_GET['type'] );

define( ROLES_APPS,  1 );  // applications
define( ROLES,       2 );  // roles themselves
define( ROLES_PRIVS, 3 );  // privileges

$role_id     = null;
$application = null;
switch( $type ) {
    case ROLES_PRIVS:
        if( !isset( $_GET['role_id'] )) die( "no valid role identifier in the request" );
        $role_id = trim( $_GET['role_id'] );
    case ROLES:
        if( !isset( $_GET['application'] )) die( "no valid application name in the request" );
        $application = trim( $_GET['application'] );
    case ROLES_APPS:
        break;
}

function app2json( $application ) {
    return json_encode(
        array (
            "label"       => '<em style="color:#0071bc;"><b>'.$application.'</b></em>',
            "expanded"    => false,
            "type"        => ROLES,
            "title"       => "applications provides a scope for roles",
            "application" => $application
        )
    );
}

function role2json( $role ) {
    return json_encode(
        array (
            "label"       => '<em style="color:black;"><b>'.$role['name'].'</b></em>',
            "expanded"    => false,
            "type"        => ROLES_PRIVS,
            "title"       => "roles encapsulate privileges",
            "application" => $role['app'],
            "name"        => $role['name'],
            "role_id"     => $role['id']
        )
    );
}

function priv2json( $priv ) {
    return json_encode( '<em style="color:#0071bc;"><b>'.$priv.'</b></em>' );
}

/*
 * Analyze and process the request
 */
try {

    $authdb = new AuthDB();
    $authdb->begin();

    // Proceed to the operation
    //
    header( "Content-type: application/json" );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    $first = true;

    if( $type == ROLES_PRIVS ) {
        $privileges = $authdb->role_privileges( $role_id );
        foreach( $privileges as $p ) {
          if( $first ) {
              $first = false;
              echo "\n".priv2json( $p );
          } else {
              echo ",\n".priv2json( $p );
          }
        }

    } else if( $type == ROLES ) {
        $roles = $authdb->roles_by_application( $application );
        foreach( $roles as $r ) {
          if( $first ) {
              $first = false;
              echo "\n".role2json( $r );
          } else {
              echo ",\n".role2json( $r );
          }
        }

    } else if( $type == ROLES_APPS ) {
        $applications = $authdb->applications();
        foreach( $applications as $a ) {
          if( $first ) {
              $first = false;
              echo "\n".app2json( $a );
          } else {
              echo ",\n".app2json( $a );
          }
        }
    } else {
        ;
    }
    print <<< HERE
 ] } }
HERE;

    $authdb->commit();

} catch( AuthDBException $e ) {
    print $e->toHtml();
}
?>
