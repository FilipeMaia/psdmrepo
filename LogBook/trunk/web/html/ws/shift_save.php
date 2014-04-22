<?php

/**
 * Save modifications made to a shift.
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $id         = $SVC->required_int ('id') ;
    $goals      = $SVC->required_str ('goals') ;
    $begin_time = $SVC->required_time('begin_time') ;
    $end_time   = $SVC->optional_time('end_time', null) ;

    if ($end_time && !$begin_time->less($end_time))
        $SVC->abort(
            'The end time of the shift must be strictly greater than '.
            'its begin time') ;

    $shift = $SVC->logbook()->find_shift_by_id($id) ;
    if (!$shift) $SVC->abort("no shift found for id={$d}") ;

    $experiment = $shift->parent() ;

    if (!$SVC->logbookauth()->canManageShifts($experiment->id()))
        $SVC->abort(
            'You are not authorized to manage shifts of the experiment') ;

    $prev_shift = $shift->previous() ;      // optional
    $next_shift = $shift->next() ;          // optional

    // Rules and restrictions for modifying the end time of the shift:
    //
    // - the shift must be closed
    // - the difference must be 1 second or greater (ignoring nanoseconds)
    // - the modifications can't propagate beyond the end time of the next shift (if any)
    //   or the present time if the next shift is still open

    if ($end_time) {
        if (!$shift->is_closed())
            $SVC->abort(
                'The end time can not be modified if the shift is still open') ;

        if (!$next_shift)
            $SVC->abort(
                'No next shift found in the database. '.
                'The operation may be used in stale context. '.
                'You may need to refresh the list of shifts and try again.') ;

        if ($end_time->toStringShort() !== $shift->end_time()->toStringShort()) {
            if ($next_shift->end_time()) {
                if (!$end_time->less($next_shift->end_time()))
                    $SVC->abort(
                        'The end time of the next shift must be strictly greater than '.
                        'the end time of the modified shift') ;
            } else {
                if (!$end_time->less(LusiTime::now()))
                    $SVC->abort(
                        'The end time of the modified shift can\'t be set in the future') ;
            }
            $shift->set_end_time($end_time) ;
            $next_shift->set_begin_time($end_time) ;
        }
    }

    // Rules and restrictions for modifying the begin time of the shift:
    //
    // - the difference must be 1 second or greater (ignoring nanoseconds)
    // - the modifications can't propagate before the begin time of the previous shift (if any)

    if ($begin_time->toStringShort() !== $shift->begin_time()->toStringShort()) {
        if ($prev_shift) {
            if (!$prev_shift->begin_time()->less($begin_time))
                $SVC->abort(
                    'The begin time of the modified shift must be strictly greater than '.
                    'the begin time of the previous shift') ;

            $prev_shift->set_end_time($begin_time) ;
        }
        $shift->set_begin_time($begin_time) ;
    }

    // Find and update the first (!) message which carries shift goals. If none exists
    // then create the new one.
    //
    // NOTE: Ignore any other messages if there are more than one of those

    $entries = $experiment->search (
        $shift->id() ,  // $shift_id=
        null ,          // $run_id=
        '' ,            // $text2search
        false ,         // $search_in_messages=
        true ,          // $search_in_tags=
        false ,         // $search_in_values=
        false ,         // $posted_at_experiment=
        true ,          // $posted_at_shifts=
        false ,         // $posted_at_runs=
        null ,          // $begin=
        null ,          // $end=
        'SHIFT_GOALS' , // $tag=
        '' ,            // $author=
        null            // $since=
    ) ;
    if (count($entries)) {
        $entry = $entries[0] ;
        $entry->update_content($entry->content_type(), $goals) ;
    } else {
        $entry = $experiment->create_entry($SVC->authdb()->authName(), 'TEXT', $goals, $shift->id()) ;
        $entry->add_tag('SHIFT_GOALS', '') ;
    }

    $SVC->finish (
        array (
            'ResultSet' => array (
                'Result' => array(
                    LogBook\LogBookUtils::shift2array (
                        $SVC->logbook()->find_shift_by_id($id)))  // -- return the updated verson of the shift
            )
        )
   ) ;
}) ;
?>
