<?php

/*
 * This script will process a request for switching to another experiment.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;

try {
    AuthDB::instance()->begin();
    RegDB::instance()->begin();

    /* Find the station number
     */
    if( !isset($_POST['station'])) {
        RegDBAuth::reporErrorHtml(
            'Missing station number among parameters of the request' );
        exit;
    }
    $station = intval($_POST['station']);

    /* Find the name of the experiment among  parameters of the request.
     */
    $experiment_param = $_POST['experiment'];
    if( !isset( $experiment_param )) {
        RegDBAuth::reporErrorHtml(
            'Missing experiment info among parameters of the request' );
        exit;
    }
    $experiment_param_split = explode( ',', $experiment_param );
    if(( count( $experiment_param_split ) < 1 ) || ( trim( $experiment_param_split[0] ) == '' )) {
        RegDBAuth::reporErrorHtml(
            'incorrectly formatted experiment parameter found in the request' );
        exit;
    }
    $experiment_name = trim( $experiment_param_split[0] );

    $experiment = RegDB::instance()->find_experiment_by_unique_name( $experiment_name );
    if( is_null( $experiment )) {
        print( RegDBAuth::reporErrorHtml(
            'No such experiment: {}'));
        exit;
    }
    $experiment_id   = $experiment->id();
    $instrument_name = $experiment->instrument()->name();

    /* Get more info on the requestor of the switch
     */
    $requestor_uid = AuthDB::authName();
    $requestor_account = RegDB::instance()->find_user_account( $requestor_uid );
    if( is_null( $requestor_account )) {
        print( RegDBAuth::reporErrorHtml(
               "Unable to get user profile for account {$requestor_uid}" ));
           exit;
    }
    $requestor_gecos = $requestor_account['gecos'];
    $requestor_email = $requestor_account['email'];

    /* Check if the authenticated user has sufficient authorization to manage
     * the switch for this or all experiments.
     */
    if( !( AuthDB::instance()->hasRole( $requestor_uid, $experiment->id(), 'ExperimentSwitch', 'Manage' ) ||
           AuthDB::instance()->hasRole( $requestor_uid, null,              'ExperimentSwitch', 'Manage' ) ||
           AuthDB::instance()->hasRole( $requestor_uid, $experiment->id(), 'ExperimentSwitch', 'Manage_'.$instrument_name ) ||
           AuthDB::instance()->hasRole( $requestor_uid, null,              'ExperimentSwitch', 'Manage_'.$instrument_name ))) {
        print( RegDBAuth::reporErrorHtml(
            'You are not authorized to manage the experiment switch'));
        exit;
    }
    
    /* Make sure the new experiment differs from the previous one. If not then simply
     * return back to the caller.
     */
    $previous_experiment_name = '';
    $previous_experiment_switch = RegDB::instance()->last_experiment_switch( $instrument_name, $station );
    if( !is_null( $previous_experiment_switch )) {
        $previous_exper_id = $previous_experiment_switch['exper_id'];
        $previous_experiment = RegDB::instance()->find_experiment_by_id( $previous_exper_id );
        if( is_null( $previous_experiment ))
            die( "fatal internal error when resolving experiment id={$previous_exper_id} in the database" );
        $previous_experiment_name = $previous_experiment->name();    
    }
    if( $previous_experiment_name != $experiment_name ) {

        /* Find persons to be notified.
         */
        $allowed_ranks = array(
            'PI'    => true,
            'IS'    => true,
            'ADMIN' => true,
            'OTHER' => true
        );
        $notify = array();
        foreach( array_keys( $_POST ) as $k => $param ) {

            if( preg_match( '/^notify_(.+)_(.+)$/', $param, $matches ) > 0 ) {

                $rank = strtoupper( $matches[1] );
                $uid  = $matches[2];

                if( !array_key_exists ( $rank, $allowed_ranks )) {
                    print( RegDBAuth::reporErrorHtml(
                        "Incorrect format of a user parameter {$param} to notify by e-mail" ));
                    exit;
                }

                /* Get user full name and their e-mail address.
                 */
                $account = RegDB::instance()->find_user_account( $uid );
                if( is_null( $account )) {
                    print( RegDBAuth::reporErrorHtml(
                        "Unknown user {$uid} found among parameters of the request" ));
                    exit;
                }
                $gecos = $account['gecos'];
                $email = $account['email'];

                array_push(
                    $notify,
                    array(
                        'uid' => $uid,
                        'gecos' => $gecos,
                        'email' => $email,
                        'rank' => $rank,
                        'notified' => 'YES'
                    )
                );
            }
        }
        
        /* Find additional instructions to be send by email to recipients
         */
        $instructions = $_POST['instructions'];


        RegDB::instance()->switch_experiment( $experiment->name(), $station, $requestor_uid, $notify );

        // TODO: Make proper adjustments to the AuthDB for the instrument's OPR account
        //
        $opr_account = strtolower($instrument_name).'opr';

        $oprelogauth2keep = array();
        foreach( array_keys( $_POST ) as $k => $param ) {
            if( preg_match( '/^oprelogauth_(.+)_(\d+)$/', $param, $matches ) > 0 ) {
                $rolename = $matches[1];
                $exper_id = $matches[2];
                array_push( $oprelogauth2keep, array( 'name' => $rolename, 'exper_id' => $exper_id  ));
            }
        }
        foreach( AuthDB::instance()->roles_by( $opr_account, 'LogBook', $instrument_name ) as $r ) {

            $rolename = $r['role']->name();
            $exper_id = $r['exper_id'];

            // Check if the role player is found in the previously built list to keep.
            // If not then get rid of that role, unless it's associated with the current
            // experiment.

            if( $exper_id == $experiment_id ) continue;

            $keep = false;
            foreach( $oprelogauth2keep as $r2keep ) {
                if(( $r2keep['name'] == $rolename ) && ( $r2keep['exper_id'] == $exper_id )) {
                    $keep = true;
                    break;
                }
            }
            if( !$keep ) {
                AuthDB::instance()->deleteRolePlayer( 'LogBook', $rolename, $exper_id, $opr_account );
            }
        }

        $oprelogauth = $_POST['oprelogauth'];
        if( isset( $oprelogauth ) && ( $oprelogauth != '' )) {
            AuthDB::instance()->createRolePlayer( 'LogBook', $oprelogauth, $experiment_id, $opr_account );
        }
        
            // Commit changes to the database only when everything has been successfully completed

            AuthDB::instance()->commit();
            RegDB::instance()->commit();

            // Send e-mail notifications.

            $instrument_experiment_names          = "{$instrument_name} / {$experiment_name}";
            $previous_experiment_instrument_names = "{$instrument_name} / {$previous_experiment_name}";

            $instructions_option = '';
            if( $instructions != '' ) {
                $instructions_option =<<<HERE
{$instructions}

__________________________________________________________________________

HERE;
        }
        function base_uri() {
            $request_uri = $_SERVER['REQUEST_URI'];
            return "https://".$_SERVER['SERVER_NAME'].substr( $request_uri, 0, strrpos( $request_uri, '/' ));
        }
        $switch_url = base_uri()."/experiment_switch.php?instr_name={$instrument_name}";

        $body =<<<HERE
{$instructions_option}
This is the automated notification on activating the following experiment:

  {$instrument_experiment_names} [ id={$experiment_id} ]

The previous experiment is now deactivated:

  {$previous_experiment_instrument_names}

The switch was requested by user:

  {$requestor_gecos} ( email: {$requestor_email} )

More information on the current experiment can be found here:

  {$switch_url}

                        ** ATTENTION **

  You've received this message either because you were identified
  as the PI of the experiment, or as an instrument scientist for '{$instrument_name}',
  or just because a person who requested the switch indicated you as
  the one whose participation is needed to accomplish the switch to
  the activated experiment. Please contact the switch requestor
  directly (see their email address at the very beginning of
  this message) in case if you think that the message was sent
  by mistake.    

HERE;
        $tmpfname = tempnam("/tmp", "experiment_switch");
        $handle = fopen( $tmpfname, "w" );
        fwrite( $handle, $body );
        fclose( $handle );

        $subject = "[ {$instrument_experiment_names} ]  experiment activated for DAQ";

        foreach( $notify as $n ) {
            $address = $n['email'];
            shell_exec( "cat {$tmpfname} | mail -s '{$subject}' {$address} -- -F 'LCLS Experiment Switch'" );
        }
        unlink( $tmpfname );
    }
    header( "Location: ../experiment_switch.php?instr_name={$instrument_name}&station={$station}" );

} catch (AuthDBException $e) { print $e->toHtml(); }
  catch (RegDBException  $e) { print $e->toHtml(); }

?>
