<?php

/*
 * Request redirector from the legacy Logbook to the Web Portal.
 */
require_once 'logbook/logbook.inc.php' ;

use LogBook\LogBook ;


function go2portal ($args) {
    if ($args) {
        $args2str = '' ;
        foreach ($args as $a) $args2str .= ($args2str ? '&' : '?').$a ;
        header("Location: ../portal/index.php{$args2str}") ;
    } else {
        header("Location: ../portal/select_experiment.php") ;        
    }
    exit ;
}

try {
    LogBook::instance()->begin() ;

    switch (trim($_GET['action'])) {

        case 'select_experiment' :
            $exper_id = intval(trim($_GET['exper_id'])) ;
            if ($exper_id)
                go2portal(array (
                    "exper_id={$exper_id}")) ;
            break ;

        case 'select_experiment_by_id' :
            $exper_id = intval(trim($_GET['id'])) ;
            if ($exper_id)
                go2portal(array (
                    "exper_id={$exper_id}")) ;
            break ;

        case 'select_experiment_and_shift' :
            $shift_id = intval(trim($_GET['shift_id'])) ;
            $shift = LogBook::instance()->find_shift_by_id($shift_id) ;
            if ($shift) {
                go2portal(array (
                    "exper_id={$shift->parent()->id()}" ,
                    "app=elog:shifts" ,
                    "shift_id={$shift->id()}")) ;
            }
            break ;

        case 'select_experiment_and_run' :
            $run_id = intval(trim($_GET['run_id'])) ;
            $run = LogBook::instance()->find_run_by_id($run_id) ;
            if ($run)
                go2portal(array (
                    "exper_id={$run->parent()->id()}" ,
                    "app=elog:runs" ,
                    "params=run:{$run->num()}")) ;
            break ;

        case 'select_run' :
            $instr_name = $_GET['instr_name'] ;
            $exper_name = $_GET['exper_name'] ;
            $experiment = LogBook::instance()->find_experiment($instr_name, $exper_name) ;
            if ($experiment) {
                $run_num = intval(trim($_GET['num'])) ;
                $run = $experiment->find_run_by_num($run_num) ;
                if ($run)
                    go2portal(array (
                        "exper_id={$experiment->id()}" ,
                        "app=elog:runs" ,
                        "params=run:{$run->num()}")) ;
            }
            break ;

        case 'select_run_by_id' :
            $run_id = intval(trim($_GET['id'])) ;
            $run = LogBook::instance()->find_run_by_id($run_id) ;
            if ($run)
                go2portal(array (
                    "exper_id={$run->parent()->id()}" ,
                    "app=elog:runs" ,
                    "params=run:{$run->num()}")) ;
            break ;

        case 'select_message' :
            $id = intval(trim($_GET['id']));
            $entry = LogBook::instance()->find_entry_by_id($id) ;
            if ($entry)
                go2portal(array (
                    "exper_id={$entry->parent()->id()}" ,
                    "app=elog:search" ,
                    "params=message:{$id}")) ;
            break ;
    }

} catch (Exception $e) { ; }

go2portal() ;

?>