<?php
/* 
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * Class LogBookFFEntry represents the API abtraction fo rfree-form entries.
 *
 * Objects of the class encapsulate all the information pertient to
 * individual entries (both initial and chained), including:
 *
 * - its unique identity in the database
 * - the context of an entry (its relevance time and experiment identity)
 * - its tags (if any)
 * - its body (inserttion time, author, body text and its type)
 * - attached documents (if any)
 * - its relationship to other entries within a discussion tree (parent entry
 *   and children, if any are present)
 *
 * @author gapon
 */
class LogBookFFEntry {

    /* Data members
     */
    private $connection;
    private $experiment;

    public $attr;

    /* Constructor
     */
    public function __construct ( $connection, $experiment, $attr ) {
        $this->connection = $connection;
        $this->experiment = $experiment;
        $this->attr = $attr;
    }

    /* Accessors
     */
    public function parent() {
        return $this->experiment; }

    public function hdr_id() {
        return $this->attr['hdr_id']; }

    public function exper_id() {
        return $this->attr['exper_id']; }

    public function relevance_time () {
        return LogBookTime::from64( $this->attr['relevance_time'] ); }

    public function id() {
        return $this->attr['id']; }

    public function parent_entry_id() {
        return $this->attr['parent_entry_id']; }

    public function insert_time () {
        return LogBookTime::from64( $this->attr['insert_time'] ); }

    public function author() {
        return $this->attr['author']; }

    public function content() {
        return $this->attr['content']; }

    public function content_type() {
        return $this->attr['content_type']; }

    /* Operations
     */
    public function children() {

        /* NOTE: We're getting only identifiers of child entries (not
         * the full entries as we want to avoid avoid potential inefficiency.
         * Also note that each identifier is packaged into an object
         * representing a full address of the entry (header identifier
         * and the entry identifier).
         */
        $list = array();

        $result = $this->connection->query(
            'SELECT id, hdr_id FROM entry WHERE parent_entry_id='.$this->id().
            ' ORDER BY id' );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            array_push(
                $list,
                new LogBookFFEntryAddr(
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }

    public function tags() {

        $list = array();

        $result = $this->connection->query(
            'SELECT t.* FROM header h, tag t WHERE h.exper_id='.$this->exper_id().
            ' AND h.id = t.hdr_id' );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            array_push(
                $list,
                new LogBookFFTag(
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }

    public function attachments() {

        /* NOTE: We aren't fetching the contents of documents with
         * this query. They will be loaded as requested through
         * the corresponding method of the attachment class.
         */
        $list = array();

        $result = $this->connection->query(
            'SELECT id,entry_id,description,document_type,  LENGTH(document) AS "document_size"'.
            ' FROM attachment WHERE entry_id='.$this->id());

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            array_push(
                $list,
                new LogBookFFAttachment(
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }
}
?>
