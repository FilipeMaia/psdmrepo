<?php

/*
 * Push e-mail nnotifications to subscribers for the file migration delays
 * 
 * For complete documentation see JIRA ticket:
 * https://jira.slac.stanford.edu/browse/PSDH-35
 *
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;
require_once 'sysmon/sysmon.inc.php' ;

use \LusiTime\LusiTime ;
use \SysMon\SysMonFileMigrDelays ;

\DataPortal\Service::run_handler ('GET', function ($SVC) {

    $now = LusiTime::now() ;

    foreach ($SVC->sysmon()->fm_delay_subscribers() as $subscr) {

        $opt = new \stdClass ;

        $opt->instr_name = $subscr->instr ? $subscr->instr : null ;
        $opt->begin_time = new LusiTime($now->sec - $subscr->last_sec) ;
        $opt->end_time   = null ;

        $DSS2FFB_not_started   = array () ;
        $DSS2FFB_not_finished  = array () ;
        $FFB2ANA_not_started   = array () ;
        $FFB2ANA_not_finished  = array () ;
        $FFB2ANA_not_started   = array () ;
        $IRODS_not_registered  = array () ;
        $ANA2HPSS_not_archived = array () ;

        $num_files = 0 ;

        foreach (SysMonFileMigrDelays::iterator ($SVC, $opt) as $f) {

            // Skip fully migrated files and detect stages delayed for longer
            // than the specified delay. Put them into the rigth category
            // to be reported later.

            $category = null ;
            if      (($f->DSS2FFB ->status == 'W') && ($f->DSS2FFB ->begin_delay >= $subscr->delay_sec)) { $category = $DSS2FFB_not_started ;   array_push($DSS2FFB_not_started,   $f) ; }
            else if (($f->DSS2FFB ->status == 'P') && ($f->DSS2FFB ->end_delay   >= $subscr->delay_sec)) { $category = $DSS2FFB_not_finished ;  array_push($DSS2FFB_not_finished,  $f) ; }
            else if (($f->FFB2ANA ->status == 'W') && ($f->FFB2ANA ->begin_delay >= $subscr->delay_sec)) { $category = $FFB2ANA_not_started ;   array_push($FFB2ANA_not_started,   $f) ; }
            else if (($f->FFB2ANA ->status == 'P') && ($f->FFB2ANA ->end_delay   >= $subscr->delay_sec)) { $category = $FFB2ANA_not_finished ;  array_push($FFB2ANA_not_finished,  $f) ; }
            else if (($f->ANA2HPSS->status == 'W') && ($f->ANA2HPSS->begin_delay >= $subscr->delay_sec)) { $category = $IRODS_not_registered ;  array_push($IRODS_not_registered,  $f) ; }
            else if (($f->ANA2HPSS->status == 'P') && ($f->ANA2HPSS->end_delay   >= $subscr->delay_sec)) { $category = $ANA2HPSS_not_archived ; array_push($ANA2HPSS_not_archived, $f) ; }

        }
        print
            "UID: ".$subscr->uid."<br>" .
            "instr: ".$opt->instr."<br>" .
            "since: ".$opt->begin_time->toStringShort()."<br>" .

            "DSS to FFB hasn't started: ".count($DSS2FFB_not_started)."<br>" .
            "DSS to FFB migration is taking for too long: ".count($DSS2FFB_not_finished)."<br>" .

            "FFB to ANA migration hasn't started: ".count($FFB2ANA_not_started)."<br>" .
            "FFB to ANA migration is taking for too long: ".count($FFB2ANA_not_finished)."<br>" .
            
            "IRODS registration still hasn't happen: ".count($IRODS_not_registered)."<br>" .
            "Files haven't been archived to HPSS for too long: ".count($ANA2HPSS_not_archived)."<br>" ;

    }
}) ;

?>
