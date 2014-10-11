<?php

/*
 * Updating an existing message
 *
 * NOTE: Wrapping the result into HTML <textarea> instead of
 * returning JSON MIME type because of the following issue:
 * 
 *   http://jquery.malsup.com/form/#file-upload
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'logbook/logbook.inc.php' ;

use LogBook\LogBookUtils ;

function handler ($SVC) {

    $id           = $SVC->required_int  ('id') ;
    $content_type = $SVC->required_str  ('content_type') ;
    $content      = $SVC->required_str  ('content') ;
    $files        = $SVC->optional_files() ;

    $entry  = $SVC->safe_assign ($SVC->logbook()->find_entry_by_id($id) ,
                                 "no e-Log entry found for for id={$id}") ;

    $experiment = $entry->parent();

    $SVC->assert ($SVC->logbookauth()->canEditMessages($experiment->id()) ,
                  'not authorized to edit messages for the experiment') ;

    $entry->update_content($content_type, $content) ;
    foreach ($files as $f)
        $attachment = $entry->attach_document($f['contents'], $f['type'], $f['description']) ;

    $experiment->notify_subscribers($entry, /* new_vs_modified = */ false) ;

    return array (
        'Entry' =>
            $entry->parent_entry_id() ?
                LogBookUtils::child2array($entry, false) :
                LogBookUtils::entry2array($entry, false)) ;
}
DataPortal\ServiceJSON::run_handler ('POST', 'handler', array('wrap_in_textarea' => true)) ;

?>
