<?php

namespace DataPortal ;

require_once 'dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

/**
 * The utility class for the Experiment Switch operations
 */
class SwitchUtils {

    /**
     * @brief Return the ready-to-export description of the current experiment
     *
     * @param DataPortal\ServiceJSON $SVC
     * @param string $instr_name
     * @param string $station
     * @return array
     */
    public static function current ($SVC, $instr_name, $station) {

        $instr = $SVC->regdb()->find_instrument_by_name($instr_name) ;
        if (!$instr) $SVC->abort("no such instrument found: '{$instr_name}'") ;

        $switch = $SVC->regdb()->last_experiment_switch($instr->name(), $station);
        if (!$switch)
            $SVC->finish(array('id' => 0 )) ;     // -- no activations for this instrument --

        $exper_id = $switch['exper_id'];

        $exper = $SVC->logbook()->find_experiment_by_id($exper_id) ;
        if (!$exper) $SVC->abort("the active experiment with id={$exper_id} is no longer registered") ;

        $first_run  = $exper->find_first_run() ;
        $last_run   = $exper->find_last_run() ;

        $first_run_begin_time = $first_run ? $first_run->begin_time() : null ;
        $first_run_end_time   = $first_run ? $first_run->end_time()   : null ;
        $last_run_begin_time  = $last_run  ? $last_run->begin_time()  : null ;
        $last_run_end_time    = $last_run  ? $last_run->end_time()    : null ;
        
        return array (

            'id'    => $exper->id() ,
            'name'  => $exper->name() ,
            'descr' => $exper->description() ,

            'contact'           => DataPortal::experiment_contact_info($exper) ,
            'decorated_contact' => DataPortal::decorated_experiment_contact_info($exper) ,
            'leader'            => $exper->leader_account() ,
            'posix_group'       => $exper->POSIX_gid() ,

            'first_run' => array (
                'num'         => $first_run            ? $first_run->num()                      : '' ,
                'begin_time'  => $first_run_begin_time ? $first_run_begin_time->toStringShort() : '' ,
                'end_time'    => $first_run_end_time   ? $first_run_end_time->toStringShort()   : '') ,

            'last_run' => array (
                'num'         => $last_run            ? $last_run->num()                      : '' ,
                'begin_time'  => $last_run_begin_time ? $last_run_begin_time->toStringShort() : '' ,
                'end_time'    => $last_run_end_time   ? $last_run_end_time->toStringShort()   : '') ,

            'switch_time'      => LusiTime::from64($switch['switch_time'])->toStringShort() ,
            'switch_requestor' => $switch['requestor_uid']
        ) ;
    }

    /**
     * @brief Send e-mail notification on the experiemnt switch event to
     *        the specified recipients.
     *
     * @param LogBookInstrument $instr
     * @param Number $station
     * @param LogBookExperiment $prev_exper
     * @param LogBookExperiment $new_exper
     * @param String $message
     * @param String $requestor_gecos
     * @param String $requestor_email
     * @param array $notify
     */
    public static function notify (
        $instr ,
        $station ,
        $prev_exper ,
        $new_exper ,
        $message ,
        $requestor_gecos ,
        $requestor_email ,
        $notify_list) {

        $instrument_experiment_names          = "{$instr->name()} / {$new_exper->name()}";
        $previous_experiment_instrument_names = "{$instr->name()} / {$prev_exper->name()}";

        $switch_url  = "https://".$_SERVER['SERVER_NAME']."/apps/experiment_switch.php?instr_name={$instr->name()}" ;

        $message_option = '' ;
        if ($message)
            $message_option =<<<HERE
{$message}

__________________________________________________________________________

HERE;

        $body =<<<HERE
{$message_option}
This is the automated notification on activating the following experiment:

  {$instrument_experiment_names} [ id={$new_exper->id()} ]

The previous experiment is now deactivated:

  {$previous_experiment_instrument_names}

The switch was requested by user:

  {$requestor_gecos} ( email: {$requestor_email} )

More information on the current experiment can be found here:

  {$switch_url}

                        ** ATTENTION **

  You've received this message either because you were identified
  as the PI of the experiment, or as an instrument scientist for '{$instr->name()}',
  or just because a person who requested the switch indicated you as
  the one whose participation is needed to accomplish the switch to
  the activated experiment. Please contact the switch requestor
  directly (see their email address at the very beginning of
  this message) in case if you think that the message was sent
  by mistake.    

HERE;
        $tmpfname = tempnam("/tmp", "experiment_switch") ;
        $handle = fopen( $tmpfname, "w") ;
        fwrite($handle, $body) ;
        fclose($handle) ;

        $subject = "[ {$instrument_experiment_names} ]  experiment activated for DAQ" ;

        foreach ($notify_list as $n) {
            $address = $n['email'] ;
            shell_exec("cat {$tmpfname} | mail -s '{$subject}' {$address} -- -F 'LCLS Experiment Switch'" ) ;
        }
        unlink( $tmpfname ) ;
    }
}

?>

