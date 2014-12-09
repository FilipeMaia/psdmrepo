<?php

namespace LogBook;

require_once( 'logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LusiTime\LusiTime;

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
    private $logbook;
    private $experiment;

    public $attr;

    /* Constructor
     */
    public function __construct ( $logbook, $experiment, $attr ) {
        $this->logbook = $logbook;
        $this->experiment = $experiment;
        $this->attr = $attr;
    }

    /* Accessors
     */
    public function parent() {
        return $this->experiment; }

    public function experiment() {
        return $this->experiment; }

    public function hdr_id() {
        return $this->attr['hdr_id']; }

    public function exper_id() {
        return $this->attr['exper_id']; }

    /**
     * Return a numeric identifier of the shift.
     * 
     * Note that this method will only return a shift which was explicitly provided
     * when posting the message. The other method LogBookFFEntry::shift() would also
     * return the default shift for the message.
     *
     * @see LogBookFFEntry::shift()
     * @return Number
     */
    public function shift_id() {
        return intval($this->attr['shift_id']); }

    public function run_id() {
        return $this->attr['run_id']; }

    public function relevance_time () {
    	/* ATTENTION: Always return the insert time of the message. No exceptions!!!
    	 */
        return $this->insert_time(); }

    public function id() {
        return $this->attr['id']; }

    public function parent_entry_id() {
        return $this->attr['parent_entry_id']; }

    public function insert_time () {
        return LusiTime::from64( $this->attr['insert_time'] ); }

    public function author() {
        return $this->attr['author']; }

    public function content() {
        return $this->attr['content']; }

    public function content_type() {
        return $this->attr['content_type']; }

    public function deleted() {
        return $this->attr['deleted_time'] != ''; }

    public function deleted_time () {
        return LusiTime::from64( $this->attr['deleted_time'] ); }

    public function deleted_by() {
        return $this->attr['deleted_by']; }

    /**
     * Find a shift object for the entry.
     * 
     * None that that would be either a shift which was explicitly specified
     * when posting the entry, or the one deduced based on the relevance time
     * of the message.
     *
     * @return LogBookShift
     * @throws LogBookException
     */
    public function shift() {
        $shift_id = intval($this->attr['shift_id']);
        $shift = $shift_id ?
            $this->parent()->find_shift_by_id( $shift_id ) :
            $this->parent()->find_shift_at   ( $this->relevance_time()->to64());
/*
        if( is_null( $shift ))
            throw new LogBookException (
                __METHOD__, "no shift for the entry. Database may be corrupted: hdr_id={$this->hdr_id()} shift_id={$shift_id}" );
*/
        return $shift;
    }

    public function run() {
        $run = $this->parent()->find_run_by_id( $this->run_id());
        if( is_null( $run ))
            throw new LogBookException (
                __METHOD__, "no run for the entry. Database may be corrupted." );
        return $run;
    }
    public function parent_entry() {
        return $this->parent_entry_id() ?
            $this->parent()->find_entry_by_id( $this->parent_entry_id()) :
            null;
    }
    public function children() {

        $list = array();

        $result = $this->logbook->query (
            "SELECT h.exper_id, h.shift_id, h.run_id, h.relevance_time, e.* FROM {$this->logbook->database}.header h, {$this->logbook->database}.entry e".
            ' WHERE h.id = e.hdr_id AND e.parent_entry_id='.$this->id().
            ' ORDER BY e.insert_time ASC' );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            array_push(
                $list,
                new LogBookFFEntry (
                    $this->logbook,
                    $this->experiment,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }

    public function create_child( $author, $content_type, $content ) {

        $subquery = "(SELECT h.id FROM {$this->logbook->database}.header h, {$this->logbook->database}.entry e WHERE h.id = e.hdr_id AND e.id=".$this->attr['id'].")";
        $this->logbook->query (
            "INSERT INTO {$this->logbook->database}.entry VALUES(NULL,".$subquery.",".$this->attr['id'].
            ",".LusiTime::now()->to64().
            ",'".$author.
            "','".$this->logbook->escape_string( $content ).
            "','".$content_type."',NULL,NULL)" );

        return $this->experiment->find_entry_by_ (
            'e.id = (SELECT LAST_INSERT_ID())' );
    }

    public function tags() {

        $list = array();

        $result = $this->logbook->query (
            "SELECT t.* FROM {$this->logbook->database}.header h, {$this->logbook->database}.tag t WHERE h.exper_id=".$this->exper_id().
            ' AND h.id = t.hdr_id'.
            ' AND h.id='.$this->hdr_id());

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            array_push (
                $list,
                new LogBookFFTag (
                    $this->logbook,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }

    public function add_tag( $tag, $value ) {

        $this->logbook->query (
            "INSERT INTO {$this->logbook->database}.tag VALUES(".$this->hdr_id().
            ",'".$this->logbook->escape_string( $tag ).
            "','".$this->logbook->escape_string( $value )."')" );

        $result = $this->logbook->query (
            "SELECT t.* FROM {$this->logbook->database}.header h, {$this->logbook->database}.tag t WHERE h.exper_id=".$this->exper_id().
            ' AND h.id = t.hdr_id'.
            ' AND h.id='.$this->hdr_id().
            " AND t.tag='".$tag."'" );

        $nrows = mysql_numrows( $result );
        if( $nrows != 1 )
            throw new LogBookException (
                __METHOD__, "unexpected size of result set" );


        return new LogBookFFTag (
            $this->logbook,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }

    public function attachments() {

        /* NOTE: We aren't fetching the contents of documents with
         * this query. They will be loaded as requested through
         * the corresponding method of the attachment class.
         */
        $list = array();

        $result = $this->logbook->query (
            'SELECT id,entry_id,description,document_type,  LENGTH(document) AS "document_size"'.
            " FROM {$this->logbook->database}.attachment WHERE entry_id=".$this->id());

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            array_push (
                $list,
                new LogBookFFAttachment (
                    $this->logbook,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }

    public function find_attachment_by_id( $id ) {

        $result = $this->logbook->query (
            'SELECT id,entry_id,description,document_type, LENGTH(document) AS "document_size"'.
            "FROM {$this->logbook->database}.attachment WHERE id=".$id );

        $nrows = mysql_numrows( $result );
        if( !$nrows ) return null;
        if( $nrows != 1 )
            throw new LogBookException (
                __METHOD__, "unexpected size of result set" );

        return new LogBookFFAttachment (
            $this->logbook,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }

    /**
     * Attach a document by the entry.
     *
     * @param string $description
     * @param string $document
     * @param string $type
     */
    public function attach_document( $document, $document_type, $description ) {

        $this->logbook->query (
            "INSERT INTO {$this->logbook->database}.attachment VALUES(NULL,".$this->id().
            ",'".$this->logbook->escape_string( $description ).
            "','".$this->logbook->escape_string( $document ).
            "','".$document_type."',NULL)" );

        return $this->find_attachment_by_id( '(SELECT LAST_INSERT_ID())' );
    }

    /**
     * Update the message contents and its type.
     *
     * @param string $description
     * @param string $document
     * @param string $type
     */
    public function update_content( $content_type, $content ) {

        $this->logbook->query (
            "UPDATE {$this->logbook->database}.entry SET".
            " content_type='".$this->logbook->escape_string( $content_type ).
            "',content='".$this->logbook->escape_string( $content ).
            "' WHERE id=".$this->id());
        $this->attr['content_type'] = $content_type;
        $this->attr['content'] = $content;
    }
}
?>
