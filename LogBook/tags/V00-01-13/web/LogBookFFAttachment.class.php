<?php
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
    private $connection;
    private $entry;

    public $attr;

    /* Constructor
     */
    public function __construct ( $connection, $entry, $attr ) {
        $this->connection = $connection;
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

        $result = $this->connection->query (
            "SELECT document FROM {$this->connection->database}.attachment WHERE id=".$this->id());

        $nrows = mysql_numrows( $result );
        if( $nrows != 1 )
            throw new LogBookException (
                __METHOD__, "unexpected size of result set" );

        return mysql_result( $result, 0 );
    }
}
?>
