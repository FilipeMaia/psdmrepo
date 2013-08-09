<?php

/*
 * This script is used as a web service to get the information about
 * an experiment from RegDB.
 *
 * TODO: Move the service to RegDB?
 */
require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBException;

if     ( isset( $_GET[ 'id'   ] )) { $exper_id   = (int) trim( $_GET[ 'id'   ] ); }
else if( isset( $_GET[ 'name' ] )) { $exper_name =       trim( $_GET[ 'name' ] ); }
else                               { die( 'no experiment identity parameter found in the requst' ); }

try {
    RegDB::instance()->begin();

    $experiment = isset( $exper_id ) ? RegDB::instance()->find_experiment_by_id         ( $exper_id   )
                                     : RegDB::instance()->find_experiment_by_unique_name( $exper_name );
    if (is_null( $experiment ))
        die( 'no such experiment found for '.( isset( $exper_id ) ? "id={$exper_id}" : "name={$exper_name}" ));

    $leader_uid     = $experiment->leader_account();
    $leader_account = RegDB::instance()->find_user_account( $leader_uid );
    $leader_gecos   = $leader_account['gecos'];
    $leader_email   = $leader_account['email'];

    $is_facility = $experiment->is_facility() ? 1 : 0;

    header( 'Content-type: application/json' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    echo json_encode (<<<HERE
{id:{$experiment->id()},name:"{$experiment->id()}",description:"{$experiment->description()}",instr_id:{$experiment->instr_id()},
instr_name:"{$experiment->instrument()->name()}",registration_time_64:{$experiment->registration_time()->to64()},registration_time:"{$experiment->registration_time()}",
begin_time_64:{$experiment->begin_time()->to64()},begin_time:"{$experiment->begin_time()}",end_time_64:{$experiment->end_time()->to64()},
end_time:"{$experiment->end_time()}",is_facility:{$is_facility},contact_info:"{$experiment->contact_info()}",posix_gid:"{$experiment->POSIX_gid()}",
leader_uid:"{$leader_uid}",leader_gecos:"{$leader_gecos}",leader_email:"{$leader_email}"}
HERE
    );

    RegDB::instance()->commit();

} catch (RegDBException $e) { print $e->toHtml(); }

?>
