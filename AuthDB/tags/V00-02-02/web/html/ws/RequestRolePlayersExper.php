<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;

use RegDB\RegDB;

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

if( !isset( $_GET['type'] )) report_error( "no valid information type in the request" );
$type = trim( $_GET['type'] );

define( 'PLAYERS_INSTR', 1 ); // instruments
define( 'PLAYERS_EXPER', 2 ); // experiments
define( 'PLAYERS',       3 ); // roles themselves

$instr_id = null;
$exper_id = null;
switch( $type ) {
    case PLAYERS:
        if( !isset( $_GET['exper_id'] )) report_error( "no valid experiment identifier in the request" );
        $exper_id = trim( $_GET['exper_id'] );
        break;
    case PLAYERS_EXPER:
        if( !isset( $_GET['instr_id'] )) report_error( "no valid instrument identifier in the request" );
        $instr_id = trim( $_GET['instr_id'] );
        break;
    case PLAYERS_INSTR:
        break;
}

function instr2json( $instr ) {
    return json_encode(
        array (
            "label"    => '<em style="color:#0071bc;"><b>'.$instr->name().'</b></em>',
            "expanded" => false,
            "type"     => PLAYERS_EXPER,
            "title"    => "select an experiment",
            "instr_id" => $instr->id()
        )
    );
}

function exper2json( $exper ) {
    return json_encode(
        array (
            "label"    => '<em style="color:#0071bc;"><b>'.$exper->name().'</b></em>',
            "expanded" => false,
            "isLeaf"   => true,
            "type"     => PLAYERS,
            "title"    => "click to see the user roles",
            "exper_id" => $exper->id(),
            "experiment" => $exper->name(),
            "instrument" => $exper->instrument()->name()
        )
    );
}

function role2json( $role ) {
    $player = $role->player();
    $group_url = $player['group'];
    if( $group_url == '' ) {
        $group_url = '';
        $user_url  = $player['user'];
        $comment = '';
    } else {
        $group_url = '<a href="javascript:view_group('."'".$group_url."'".')">'.$group_url.'</a>';
        $user_url  = '';
        $comment = 'all members of the group;';
    }
    if( is_null( $role->exper_id())) {
    	if( $comment != '' ) $comment .= '<br>';
    	$comment .= 'also accross all experiments;';
    }

    return json_encode(
        array (
            "application" => $role->application(),
            "role"        => $role->name(),
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

    RegDB::instance()->begin();
    AuthDB::instance()->begin();

    print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;

    $first = true;

    if( $type == PLAYERS_INSTR ) {
        foreach (RegDB::instance()->instruments() as $i) {
            if( $first ) {
                $first = false;
                echo "\n".instr2json( $i );
            } else {
                echo ",\n".instr2json( $i );
            }
        }
    } else if( $type == PLAYERS_EXPER ) {
        $instrument = RegDB::instance()->find_instrument_by_id( $instr_id )
            or report_error("No such instrument");
        foreach (RegDB::instance()->experiments_for_instrument($instrument->name()) as $e) {
          if( $first ) {
              $first = false;
              echo "\n".exper2json( $e );
          } else {
              echo ",\n".exper2json( $e );
          }
        }
    } else if( $type == PLAYERS ) {
        foreach (AuthDB::instance()->roles( $exper_id ) as $r) {
            // Skip roles which are known not to be associated with particular experiments
            //
            // TODO: This is a request from Andy Salnikov.
            //
            if(( $r->application() == 'RoleDB' ) || ( $r->application() == 'RegDB' )) {
                ;
            } else {
                if( $first ) {
                    $first = false;
                    echo "\n".role2json( $r );
                } else {
                    echo ",\n".role2json( $r );
                }
            }
        }
    }
    print <<< HERE
 ] } }
HERE;

    RegDB::instance()->commit();
    AuthDB::instance()->commit();

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }


?>
