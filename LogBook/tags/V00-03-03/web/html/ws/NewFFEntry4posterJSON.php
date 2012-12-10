<?php

/*
 * This script will process a request from e-log Poster for creating new free-form
 * entry in the specified scope.
 *
 * Parameters:
 * 
 * Rasult:
 * 
 *   status      - always
 *   message     - if failed
 *   updated     - if successful
 *   message_id  - if successful
 */

require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LogBook\LogBookUtils ;

use LusiTime\LusiTime ;


\DataPortal\ServiceJSON::run_handler ('POST', function ($SVC) {

    $instrument_name = trim($SVC->required_str('instrument')) ;
    $experiment_name = trim($SVC->required_str('experiment')) ;
    $message_text    =      $SVC->required_str('message_text') ;
    $author_account  = trim($SVC->required_str('author_account')) ;
    $scope           = trim($SVC->required_str('scope')) ;

    $shift_id          = null ;
    $run_id            = null ;
    $run_num           = null ;
    $parent_message_id = null ;

    switch ($scope) {
        case 'experiment' :
            break ;
        case 'shift' :
            $shift_id = $SVC->required_int('shift_id') ;
            break ;
        case 'run' :
            $run_id = $SVC->optional_int('run_id', null) ;
            if (is_null($run_id)) $run_num = $SVC->required_int('run_num') ;
            break ;
        case 'message' :
            $parent_message_id = $SVC->required_int('message_id') ;
            break ;
        default :
            $SVC->abort("unsupported scope: '{$scope}'") ;
    }

    $relevance_time =      $SVC->optional_time('relevance_time', LusiTime::now()) ;
    $text4child     = trim($SVC->optional_str ('text4child',     '')) ;

    $tags = array () ;
    for ( $i=0 ; $i < $SVC->optional_int('num_tags', 0) ; $i++) {
        $tag = trim($SVC->optional_str('tag_name_'.$i, '')) ;
        if ($tag !== '')
                array_push (
                    $tags ,
                    array (
                        'tag'   => $tag ,
                        'value' => trim($SVC->required_str('tag_value_'.$i)))) ;
    }

    $files = array () ;
    foreach (array_keys($_FILES) as $file_key) {

        $name  = $_FILES[$file_key]['name'] ;
        $error = $_FILES[$file_key]['error'] ;

        if ($error != UPLOAD_ERR_OK) {
            if ($error == UPLOAD_ERR_NO_FILE) continue ;
            $SVC->abort (
                "Attachment '{$name}' couldn't be uploaded because of the following problem: '".
                LogBookUtils::upload_err2string($error)."'.") ;
        }
        if ($name) {

            // Read file contents into a local variable
            //
            $location = $_FILES[$file_key]['tmp_name'] ;
            $fd = fopen($location, 'r') or $SVC->abort("failed to open file: '{$location}'") ;
            $contents = fread($fd, filesize($location )) ;
            fclose($fd) ;

            // Get its description. If none is present then use the original
            // name of the file at client's side.
            //
            $description = trim($SVC->optional_str($file_key, $name)) ;

            array_push (
                $files ,
                array (
                    'type'        => $_FILES[$file_key]['type'] ,
                    'description' => $description ,
                    'contents'    => $contents)) ;
        }
    }

    $content_type = "TEXT" ;

    $experiment = $SVC->logbook()->find_experiment($instrument_name, $experiment_name) or
        $SVC->abort('no such experiment') ;

    $SVC->logbookauth()->canPostNewMessages($experiment->id(), $author_account) or
        $SVC->abort('You are not authorized to post messages for the experiment') ;

    if (($scope == 'run') && is_null($run_id)) {
        $run = $experiment->find_run_by_num($run_num) ;
        if (is_null($run)) {
            $first_run = $experiment->find_first_run() ;
            $last_run  = $experiment->find_last_run() ;
            $SVC->abort(
                (is_null($first_run) || is_null($last_run)) ?
                    "No runs have been taken by this experiment yet." :
                    "Run number {$run_num} has not been found. Allowed range of runs is: {$first_run->num()}..{$last_run->num()}.") ;
        }
        $run_id = $run->id() ;
    }

    /* If the request has been made in a scope of some parent entry then
     * one the one and create the new one in its scope.
     *
     * NOTE: Remember that child entries have no tags, but they
     *       are allowed to have attachments.
     */
    if ($scope === 'message') {
        $parent = $experiment->find_entry_by_id($parent_message_id) or $SVC->abort('no such parent message exists') ;
        $entry  = $parent->create_child($author_account, $content_type, $message_text) ;
    } else {
        $entry = $experiment->create_entry($author_account, $content_type, $message_text, $shift_id, $run_id, $relevance_time) ;
        foreach ($tags as $t)
            $entry->add_tag($t['tag'], $t['value']) ;
    }
    foreach ($files as $f)
        $entry->attach_document($f['contents'], $f['type'], $f['description']) ;

    if ($text4child !== '')
        $entry = $entry->create_child($author_account, $content_type, $text4child) ;

    $experiment->notify_subscribers ($entry) ;

    $message_id = $entry->id() ;

    $SVC->finish(array ('message_id' => $message_id)) ;
}) ;

?>
