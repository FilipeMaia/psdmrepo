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

// Uncomment this to trigget reporting for any file found in scopes
// requested by subscribers.
//
// define ('DEBUG', 1) ;

\DataPortal\Service::run_handler ('GET', function ($SVC) {

    $now = LusiTime::now() ;
    
    foreach ($SVC->sysmon()->fm_delay_subscribers() as $subscr) {

        $event_names = array() ;
        $event2descr = array() ;
        $event2files = array() ;
        foreach ($subscr->events() as $event) {
            array_push($event_names, $event->name) ;
            $event2descr[$event->name] = $event->descr ;
            $event2files[$event->name] = array() ;
        }

        $opt = new \stdClass ;

        $opt->instr_name = $subscr->instr ? $subscr->instr : null ;
        $opt->begin_time = new LusiTime($now->sec - $subscr->last_sec) ;
        $opt->end_time   = null ;
        
if (defined('DEBUG')) {
        $count = 0 ;
}
        $fileitr = SysMonFileMigrDelays::iterator ($SVC, $opt) ;
        foreach ($fileitr as $f) {

if (defined('DEBUG')) {
            $subscr->delay_sec = 0 ;
            switch ($count++ % 6) {
                case 0: $f->DSS2FFB ->status = 'W' ; break ;
                case 1: $f->DSS2FFB ->status = 'P' ; break ;
                case 2: $f->FFB2ANA ->status = 'W' ; break ;
                case 3: $f->FFB2ANA ->status = 'P' ; break ;
                case 4: $f->ANA2HPSS->status = 'W' ; break ;
                case 5: $f->ANA2HPSS->status = 'P' ; break ;
            }
}

            // Skip fully migrated files and detect stages delayed for longer
            // than the specified delay. Put them into the rigth category
            // to be reported later.

            if (array_key_exists('xtc.DSS2FFB.begin', $event2descr)) {
                $event =         'xtc.DSS2FFB.begin' ;
                $stage = $f->DSS2FFB ;
                $delay = $stage->begin_delay ;
                if (($stage->status == 'W') && ($delay >= $subscr->delay_sec)) {
                    array_push($event2files[$event], array (
                        'file'  => $f ,
                        'delay' => $delay ,
                        'host'  => $stage->host)) ;
                    continue ;
                }
            }
            if (array_key_exists('xtc.DSS2FFB.end', $event2descr)) {
                $event =         'xtc.DSS2FFB.end' ;
                $stage = $f->DSS2FFB ;
                $delay = $stage->end_delay ;
                if (($stage->status == 'P') && ($delay >= $subscr->delay_sec)) {
                    array_push($event2files[$event], array (
                        'file'  => $f ,
                        'delay' => $delay ,
                        'host'  => $stage->host)) ;
                    continue ;
                }
            }
            if (array_key_exists('xtc.FFB2ANA.begin', $event2descr)) {
                $event =         'xtc.FFB2ANA.begin' ;
                $stage = $f->FFB2ANA ;
                $delay = $stage->begin_delay ;
                if (($stage->status == 'W') && ($delay >= $subscr->delay_sec)) {
                    array_push($event2files[$event], array (
                        'file'  => $f ,
                        'delay' => $delay ,
                        'host'  => $stage->host)) ;
                    continue ;
                }
            }
            if (array_key_exists('xtc.FFB2ANA.end', $event2descr)) {
                $event =         'xtc.FFB2ANA.end' ;
                $stage = $f->FFB2ANA ;
                $delay = $stage->end_delay ;
                if (($stage->status == 'P') && ($delay >= $subscr->delay_sec)) {
                    array_push($event2files[$event], array (
                        'file'  => $f ,
                        'delay' => $delay ,
                        'host'  => $stage->host)) ;
                    continue ;
                }
            }
            if (array_key_exists('xtc.ANA2HPSS.begin', $event2descr)) {
                $event =         'xtc.ANA2HPSS.begin' ;
                $stage = $f->ANA2HPSS ;
                $delay = $stage->begin_delay ;
                if (($stage->status == 'W') && ($delay >= $subscr->delay_sec)) {
                    array_push($event2files[$event], array (
                        'file'  => $f ,
                        'delay' => $delay ,
                        'host'  => $stage->host)) ;
                    continue ;
                }
            }
            if (array_key_exists('xtc.ANA2HPSS.end', $event2descr)) {
                $event =         'xtc.ANA2HPSS.end' ;
                $stage = $f->ANA2HPSS ;
                $delay = $stage->end_delay ;
                if (($stage->status == 'P') && ($delay >= $subscr->delay_sec)) {
                    array_push($event2files[$event], array (
                        'file'  => $f ,
                        'delay' => $delay ,
                        'host'  => $stage->host)) ;
                    continue ;
                }
            }
        }
        $exper_id2name = $fileitr->experiment_names() ;

        $msg = '' ;
        foreach ($event_names as $event) {
            if (count($event2files[$event])) {
                if (!$msg) {
                    $url = 'https://pswww.slac.stanford.edu/' ;
                    $msg = <<<HERE
Greetings, user '{$subscr->uid}'!

This automated notification  message was set to  you  by the Data Management
System Monitoring in order to  let you  know  about  file  migration  delays
matching  your  subscription  criteria:

   Files of instrument '{$subscr->instr}' (all if empty)
   Most recent files created less than {$subscr->last_sec} seconds ago

You can  manage  your  criteria  or unsubscribe from recieving notifications
by  using "Data Migration Monitor/Notifier"  found  in the "Data Management"
section of: {$url}

Delayed files are groupped by categories:

HERE;
                }
                $msg .= <<<HERE

    {$event2descr[$event]}

        Experiment |  Run |                 File |             Host | Delay [s]
        -----------+------+----------------------+------------------+-----------

HERE;
                foreach ($event2files[$event] as $f) {
                    $msg .= sprintf (
                        "        %10s | %4s | %20s | %16s | %9d\n" ,
                        $exper_id2name[$f['file']->exper_id] ,
                        $f['file']->run ,
                        $f['file']->name ,
                        $f['host'] ,
                        $f['delay']) ;
                }
            }
        }
        if ($msg)
            $SVC->configdb()->do_notify (
                "{$subscr->uid}@slac.stanford.edu" ,
                "*** ALERT ***" ,
                $msg ,
                'LCLS Data Migration Monitor') ;
    }
}) ;

?>
