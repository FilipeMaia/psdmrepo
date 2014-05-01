<?php

/*
 * Create a new message
 *
 * NOTE: Wrapping the result into HTML <textarea> instead of
 * returning JSON MIME type because of the following issue:
 * 
 *   http://jquery.malsup.com/form/#file-upload
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LogBook\LogBookUtils ;

use LusiTime\LusiTime ;

function handler ($SVC) {

    $id           = $SVC->required_int('id') ;
    $message      = $SVC->required_str('message_text') ;
    $author       = $SVC->required_str('author_account') ;
    $scope        = $SVC->required_str('scope') ;

    $shift_id     = $SVC->optional_int('shift_id',   null) ;
    $run_id       = $SVC->optional_int('run_id',     null) ;
    $run_num      = $SVC->optional_int('run_num',    null) ;
    $message_id   = $SVC->optional_int('message_id', null) ;

    switch ($scope) {
        case 'shift'      : $SVC->assert ($shift_id,           'no valid shift id in the request') ; break ;
        case 'run'        : $SVC->assert ($run_id || $run_num, 'no valid run id or number in the request') ; break ;
        case 'message'    : $SVC->assert ($message_id,         'no valid id for the parent message in the request') ; break ;
        case 'experiment' : break ;
        default :
            $SVC->assert (false, "invalid scope: '{$scope}'") ;
    }

    $relevance_time = $SVC->optional_time('relevance_time', LusiTime::now()) ;
    $files          = $SVC->optional_files() ;
    $tags           = $SVC->optional_tags() ;

    // If set this will tell the script to return the updated parent object
    // instead of the new one (the reply).

    $return_parent = $SVC->optional_bool('return_parent', false) ;


    $experiment = $SVC->safe_assign ($SVC->logbook()->find_experiment_by_id($id) ,
                                     'no such experiment') ;

    $SVC->assert ($SVC->logbookauth()->canPostNewMessages($experiment->id()) ,
                  'not authorized to post messages for the experiment') ;

    if (($scope == 'run') && is_null($run_id)) {
    	$run = $experiment->find_run_by_num($run_num) ;
    	if (!$run) {
    	    $first_run = $experiment->find_first_run() ;
    	    $last_run  = $experiment->find_last_run() ;
            $SVC->abort (!$first_run || !$last_run ?
                         "No runs have been taken by this experiment yet." :
                         "Run number {$run_num} has not been found. Allowed range of runs is: {$first_run->num()}..{$last_run->num()}.") ;
    	}
    	$run_id = $run->id() ;
    }

    $content_type = "TEXT" ;

    /* If the request has been made in a scope of some parent entry then
     * one the one and create the new one in its scope.
     *
     * NOTE: Remember that child entries have no tags, but they
     *       are allowed to have attachments.
     */
    if ($scope == 'message') {
        $parent = $SVC->safe_assign ($experiment->find_entry_by_id($message_id) ,
                                     "no such parent message exists") ;
        $entry = $parent->create_child($author, $content_type, $message) ;
    } else {
        $entry = $experiment->create_entry($author, $content_type, $message, $shift_id, $run_id, $relevance_time) ;        
        foreach($tags as $t)
            $tag = $entry->add_tag($t['tag'], $t['value']) ;
    }
    foreach ($files as $f)
        $attachment = $entry->attach_document($f['contents'], $f['type'], $f['description']) ;

    $experiment->notify_subscribers($entry) ;

    return array (
        'Entry' =>
            $scope == 'message' ?
                LogBookUtils::child2array($return_parent ? $parent : $entry, false) :
                LogBookUtils::entry2array($entry, false)) ;
}
DataPortal\ServiceJSON::run_handler ('POST', 'handler', array('wrap_in_textarea' => true)) ;


?>
