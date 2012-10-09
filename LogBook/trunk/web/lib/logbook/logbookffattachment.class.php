<?php

namespace LogBook;

require_once( 'logbook.inc.php' );

/* 
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * Class LogBookFFAttachment is an abstraction for documents attached
 * by free-form entries.
 *
 * @author gapon
 */
class LogBookFFAttachment {

    /* Data members
     */
    private $logbook;
    private $entry;

    public $attr;

    /* Constructor
     */
    public function __construct ( $logbook, $entry, $attr ) {
        $this->logbook = $logbook;
        $this->entry = $entry;
        $this->attr = $attr;
    }

    /* Accessors
     */
    public function parent() {
        return $this->entry; }

    public function id() {
        return $this->attr['id']; }

    public function entry_id() {
        return $this->attr['entry_id']; }

    public function description() {
        return $this->attr['description']; }

    public function document_type() {
        return $this->attr['document_type']; }

    public function document_size() {
        return $this->attr['document_size']; }

    public function document() {

        $result = $this->logbook->query (
            "SELECT document FROM {$this->logbook->database}.attachment WHERE id=".$this->id());

        $nrows = mysql_numrows( $result );
        if( $nrows != 1 )
            throw new LogBookException (
                __METHOD__, "unexpected size of result set" );

        return mysql_result( $result, 0 );
    }
    public function document_preview() {

        $result = $this->logbook->query (
            "SELECT document_preview FROM {$this->logbook->database}.attachment WHERE id=".$this->id());

        $nrows = mysql_numrows( $result );
        if( $nrows != 1 )
            throw new LogBookException (
                __METHOD__, "unexpected size of result set" );

        return mysql_result( $result, 0 );
    }
    public function update_document_preview( $preview ) {

        $this->logbook->query (
            "UPDATE {$this->logbook->database}.attachment SET".
            " document_preview='".$this->logbook->escape_string( $preview ).
            "' WHERE id=".$this->id());
    }
}
?>
