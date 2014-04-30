<?php

require_once('authdb/authdb.inc.php');

use AuthDB\AuthDB;

/*
 * This script will process requests for various information stored in the database.
 * The result will be returned as JSON object (array).
 */
header( "Content-type: application/json" );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

function report_error($msg) {
    print json_encode ( array (
        "ResultSet" => array("Result" => array()),
        "Message" => $msg
    ));
    exit;
}

if(!isset($_GET['type'])) report_error("no valid information type in the request");
$type = trim($_GET['type']);

define('ROLES_APPS',  1);  // applications
define('ROLES',       2);  // roles themselves
define('ROLES_PRIVS', 3);  // privileges

$role_id     = null;
$application = null;
switch($type) {
    case ROLES_PRIVS:
        if(!isset($_GET['role_id'])) report_error("no valid role identifier in the request");
        $role_id = trim($_GET['role_id']);
    case ROLES:
        if(!isset($_GET['application'])) report_error("no valid application name in the request");
        $application = trim($_GET['application']);
    case ROLES_APPS:
        break;
}

function app2json($application) {
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

function role2json($role) {
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

function priv2json($priv) {
    return json_encode('<em style="color:#0071bc;"><b>'.$priv.'</b></em>');
}

/*
 * Analyze and process the request
 */
try {

    AuthDB::instance()->begin();

    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    $first = true;

    if ($type == ROLES_PRIVS) {
        foreach (AuthDB::instance()->role_privileges($role_id) as $p) {
            if ($first) {
                $first = false;
                echo "\n".priv2json($p);
            } else {
                echo ",\n".priv2json($p);
            }
        }
    } elseif ($type == ROLES) {
        foreach (AuthDB::instance()->roles_by_application($application) as $r) {
            if ($first) {
                $first = false;
                echo "\n".role2json($r);
            } else {
                echo ",\n".role2json($r);
            }
        }
    } elseif ($type == ROLES_APPS) {
        foreach (AuthDB::instance()->applications() as $a) {
            if ($first) {
                $first = false;
                echo "\n".app2json($a);
            } else {
                echo ",\n".app2json($a);
            }
        }
    }
    print <<< HERE
 ] } }
HERE;

    AuthDB::instance()->commit();

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>
