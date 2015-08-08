<?php

/*
 * Extend an existing message.
 *
 * NOTE: Wrapping the result into HTML <textarea> instead of
 * returning JSON MIME type because of the following issue:
 * 
 *   http://jquery.malsup.com/form/#file-upload
 */
require_once 'dataportal/dataportal.inc.php' ;

function handler ($SVC) {

    $message_id = $SVC->required_int  ('message_id') ;
    $files      = $SVC->optional_files() ;
    $tags       = $SVC->optional_tags () ;

    $entry = $SVC->safe_assign ($SVC->logbook()->find_entry_by_id($message_id) ,
                                "no e-Log entry found for for id={$message_id}") ;

    $experiment = $entry->parent() ;

    $SVC->assert ($SVC->logbookauth()->canPostNewMessages($experiment->id()) ,
                  'not authorized to extend messages for this experiment') ;

    $extended_tags = array ();
    foreach ($tags as $t) {
        $tag = $entry->add_tag($t['tag'], $t['value']) ;
        array_push (
        	$extended_tags,
        	array (
                    "tag"   => $tag->tag() ,
                    "value" => $tag->value())) ;
    }

    $extended_attachments = array () ;
    foreach ($files as $f) {
        $attachment = $entry->attach_document($f['contents'], $f['type'], $f['description']) ;
        array_push (
        	$extended_attachments ,
        	array (
                    "id"          => $attachment->id() ,
                    "type"        => $attachment->document_type() ,
                    "size"        => $attachment->document_size() ,
                    "description" => $attachment->description() ,
                    "url"         => '<a href="attachments/'.$attachment->id().'/'.$attachment->description().'" target="_blank" class="lb_link">'.$attachment->description().'</a>')) ;
    }
    return array (
        'Extended' => array (
            'attachments' => $extended_attachments ,
            'tags'        => $extended_tags)) ; 
}
DataPortal\ServiceJSON::run_handler ('POST', 'handler', array('wrap_in_textarea' => true)) ;

?>
