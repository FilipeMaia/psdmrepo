<?php

require_once( 'AuthDB/AuthDB.inc.php' );
require_once( 'RegDB/RegDB.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use RegDB\RegDB;
use RegDB\RegDBException;

/*
 * This script will process requests for various information stored in the database.
 * The result will be returned as JSON object (array).
 */
if( !isset( $_GET['role_id'] )) die( "no valid role identifier in the request" );
$role_id = trim( $_GET['role_id'] );

/*
function player2json( $player ) {
    $group_url = $player['group'];
    if( $group_url == '' ) {
        $group_url = '';
        $user_url  = $player['user'];
    } else {
        $group_url = '<a href="javascript:view_group('."'".$group_url."'".')">'.$group_url.'</a>';
        $user_url  = '';
    }
    return json_encode(
        array (
            "instrument" => $player['instrument'],
            "experiment" => $player['experiment'],
            "group"      => $group_url,
            "user"       => $user_url,
            "comment"    => $player['comment']
        )
    );
}
*/

function role2json( $role, $regdb ) {
    $player = $role->player();
    $group_url = $player['group'];
    if( $group_url == '' ) {
        $group_url = '';
        $user_url  = $player['user'];
        $comment = '';
    } else {
        $group_url = '<a href="javascript:view_group('."'".$group_url."'".')">'.$group_url.'</a>';
        $user_url  = '';
        $comment = 'all members of the group';
    }

    $instrument_url = '&lt;any&gt;';
    $experiment_url = '&lt;any&gt;';
    $exper_id = $role->exper_id();
    if( !is_null( $exper_id )) {
        $experiment = $regdb->find_experiment_by_id( $exper_id )
            or die( "no experiment with id={$exper_id} found." );
        $instrument_url = $experiment->instrument()->name();
        $experiment_url = $experiment->name();
    }

    return json_encode(
        array (
            "instrument"  => $instrument_url,
            "experiment"  => $experiment_url,
            "group"       => $group_url,
            "user"        => $player['user'],
            "comment"     => $comment
        )
    );
}

/*
 * Analyze and process the request
 */
try {

    $regdb = new RegDB();
    $regdb->begin();

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

    $roles = $authdb->roles_by_id( $role_id );
    foreach( $roles as $r ) {
        if( $first ) {
            $first = false;
            echo "\n".role2json( $r, $regdb );
        } else {
            echo ",\n".role2json( $r, $regdb );
        }
    }
/*


    $players = array(
        array(
            "instrument" => "AMO",
            "experiment" => "Installation",
            "group"      => "lab-admin",
            "user"       => "",
            "comment"    => "all members of the group"
        ),
        array(
            "instrument" => "AMO",
            "experiment" => "Installation",
            "group"      => "",
            "user"       => "gapon"
        ),
        array(
            "instrument" => "AMO",
            "experiment" => "Comissioning",
            "group"      => "xr",
            "user"       => "",
            "comment"    => "all members of the group"
        ),
        array(
            "instrument" => "AMO",
            "experiment" => "Comissioning",
            "group"      => "",
            "user"       => "perazzo"
        )
    );
    foreach( $players as $p ) {
        if( $first ) {
            $first = false;
            echo "\n".player2json( $p );
        } else {
            echo ",\n".player2json( $p );
        }
    }
*/
    print <<< HERE
 ] } }
HERE;

    $regdb->commit();
    $authdb->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
} catch( AuthDBException $e ) {
    print $e->toHtml();
}
?>
