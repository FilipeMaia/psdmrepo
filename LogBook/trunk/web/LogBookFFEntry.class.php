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

    public function shift_id() {
        return $this->attr['shift_id']; }

    public function run_id() {
        return $this->attr['run_id']; }

    public function relevance_time () {
        return is_null( $this->attr['relevance_time'] ) ?
            null :
            LusiTime::from64( $this->attr['relevance_time'] ); }

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

    /* Operations
     */
    public function shift() {
        $shift = $this->parent()->find_shift_by_id( $this->shift_id());
        if( is_null( $shift ))
            throw new LogBookException (
                __METHOD__, "no shift for the entry. Database may be corrupted." );
        return $shift;
    }

    public function run() {
        $run = $this->parent()->find_run_by_id( $this->run_id());
        if( is_null( $run ))
            throw new LogBookException (
                __METHOD__, "no run for the entry. Database may be corrupted." );
        return $run;
    }

    public function children() {

        $list = array();

        $result = $this->connection->query (
            "SELECT h.exper_id, h.shift_id, h.run_id, h.relevance_time, e.* FROM {$this->connection->database}.header h, {$this->connection->database}.entry e".
            ' WHERE h.id = e.hdr_id AND e.parent_entry_id='.$this->id().
            ' ORDER BY e.insert_time ASC' );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            array_push(
                $list,
                new LogBookFFEntry (
                    $this->connection,
                    $this->experiment,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }

    public function create_child( $author, $content_type, $content ) {

        $subquery = "(SELECT h.id FROM {$this->connection->database}.header h, {$this->connection->database}.entry e WHERE h.id = e.hdr_id AND e.id=".$this->attr['id'].")";
        $this->connection->query (
            "INSERT INTO {$this->connection->database}.entry VALUES(NULL,".$subquery.",".$this->attr['id'].
            ",".LusiTime::now()->to64().
            ",'".$author.
            "','".$this->connection->escape_string( $content ).
            "','".$content_type."')" );

        return $this->experiment->find_entry_by_ (
            'e.id = (SELECT LAST_INSERT_ID())' );
    }

    public function tags() {

        $list = array();

        $result = $this->connection->query (
            "SELECT t.* FROM {$this->connection->database}.header h, {$this->connection->database}.tag t WHERE h.exper_id=".$this->exper_id().
            ' AND h.id = t.hdr_id'.
            ' AND h.id='.$this->hdr_id());

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            array_push (
                $list,
                new LogBookFFTag (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }

    public function add_tag( $tag, $value ) {

        $this->connection->query (
            "INSERT INTO {$this->connection->database}.tag VALUES(".$this->hdr_id().
            ",'".$this->connection->escape_string( $tag ).
            "','".$this->connection->escape_string( $value )."')" );

        $result = $this->connection->query (
            "SELECT t.* FROM {$this->connection->database}.header h, {$this->connection->database}.tag t WHERE h.exper_id=".$this->exper_id().
            ' AND h.id = t.hdr_id'.
            ' AND h.id='.$this->hdr_id().
            " AND t.tag='".$tag."'" );

        $nrows = mysql_numrows( $result );
        if( $nrows != 1 )
            throw new LogBookException (
                __METHOD__, "unexpected size of result set" );


        return new LogBookFFTag (
            $this->connection,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }

    public function attachments() {

        /* NOTE: We aren't fetching the contents of documents with
         * this query. They will be loaded as requested through
         * the corresponding method of the attachment class.
         */
        $list = array();

        $result = $this->connection->query (
            'SELECT id,entry_id,description,document_type,  LENGTH(document) AS "document_size"'.
            " FROM {$this->connection->database}.attachment WHERE entry_id=".$this->id());

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            array_push (
                $list,
                new LogBookFFAttachment (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }

    public function find_attachment_by_id( $id ) {

        $result = $this->connection->query (
            'SELECT id,entry_id,description,document_type, LENGTH(document) AS "document_size"'.
            "FROM {$this->connection->database}.attachment WHERE id=".$id );

        $nrows = mysql_numrows( $result );
        if( !$nrows ) return null;
        if( $nrows != 1 )
            throw new LogBookException (
                __METHOD__, "unexpected size of result set" );

        return new LogBookFFAttachment (
            $this->connection,
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

        $this->connection->query (
            "INSERT INTO {$this->connection->database}.attachment VALUES(NULL,".$this->id().
            ",'".$this->connection->escape_string( $description ).
            "','".$this->connection->escape_string( $document ).
            "','".$document_type."')" );

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

        $this->connection->query (
            "UPDATE {$this->connection->database}.entry SET".
            " content_type='".$this->connection->escape_string( $content_type ).
            "',content='".$this->connection->escape_string( $content ).
            "' WHERE id=".$this->id());
        $this->attr['content_type'] = $content_type;
        $this->attr['content'] = $content;
    }
}
?>
