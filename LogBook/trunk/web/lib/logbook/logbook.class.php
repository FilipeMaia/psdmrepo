<?php

namespace LogBook;

require_once( 'filemgr/filemgr.inc.php' );
require_once( 'logbook.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use FileMgr\DbConnection;

use RegDB\RegDB;

class LogBook  extends DbConnection {

    private static $instance = null;

    /**
     * Return an instance of the object initialzied with default version
     * of parameters.
     */
    public static function instance() {
        if( is_null(LogBook::$instance)) LogBook::$instance =
            new LogBook(
                LOGBOOK_DEFAULT_HOST,
                LOGBOOK_DEFAULT_USER,
                LOGBOOK_DEFAULT_PASSWORD,
                LOGBOOK_DEFAULT_DATABASE
            );
        return LogBook::$instance;
    }
	

    /* Constructor
     *
     * Construct the top-level API object using the specified connection
     * parameters. Put null to envorce default values of parameters.
     */
    public function __construct ($host, $user, $password, $database) {
        parent::__construct ( $host, $user, $password, $database );
    }

    public function regdb() {
    	RegDB::instance()->begin();
    	return RegDB::instance();
    }

    /* ===============
     *   INSTRUMENTS
     * ===============
     */
    public function instruments () {
        RegDB::instance()->begin();
        return RegDB::instance()->instruments();
    }

    /* ===============
     *   EXPERIMENTS
     * ===============
     */
    public function experiments ( $condition = '' ) {

        $list = array();

        RegDB::instance()->begin();
        foreach (RegDB::instance()->experiments() as $e )
            array_push(
                $list,
                new LogBookExperiment ($this, $e));

        return $list;
    }
    public function experiments_for_instrument ( $name ) {

        $list = array();

        RegDB::instance()->begin();
        foreach (RegDB::instance()->experiments_for_instrument( $name ) as $e )
            array_push(
                $list,
                new LogBookExperiment ($this, $e));

        return $list;
    }

    public function find_experiment_by_id ( $id ) {
        RegDB::instance()->begin();
        $e = RegDB::instance()->find_experiment_by_id( $id ) ;
        return is_null( $e ) ? null : new LogBookExperiment ($this, $e) ;
    }

    public function find_experiment ( $instrument_name, $experiment_name ) {
        RegDB::instance()->begin();
        $e = RegDB::instance()->find_experiment( $instrument_name, $experiment_name ) ;
        return is_null( $e ) ? null : new LogBookExperiment ($this, $e) ;
    }

    public function find_experiment_by_name ( $name ) {
        RegDB::instance()->begin();
        $e = RegDB::instance()->find_experiment_by_name( $name ) ;
        return is_null( $e ) ?
            null : new LogBookExperiment ($this, $e) ;
    }

    /* ============================
     *   ID-BASED DIRECT LOCATORS
     * ============================
     */
    public function find_attachment_by_id( $id ) {

        $result = $this->query (
            "SELECT h.exper_id, a.entry_id FROM {$this->database}.header h, {$this->database}.entry e, {$this->database}.attachment a".
            " WHERE a.id=".$id.
            " AND h.id=e.hdr_id AND e.id=a.entry_id" );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 ) {

            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );

            $exper_id = $attr['exper_id'];
            $experiment = $this->find_experiment_by_id( $exper_id );
            if( is_null( $experiment ))
                throw new LogBookException(
                    __METHOD__,
                    "no experiment for id: {$exper_id} in database. Database can be corrupted." );

            $entry_id = $attr['entry_id'];
            $entry = $experiment->find_entry_by_id( $entry_id );
            if( is_null( $entry ))
                throw new LogBookException(
                    __METHOD__,
                    "no free-form entry for id: {$entry_id} in database. Database can be corrupted." );

            return $entry->find_attachment_by_id( $id );
        }
        return null;
    }

    public function find_entry_by_id( $id ) {

        // Find an experiment the shift belongs to
        //
        $result = $this->query (
            "SELECT h.exper_id, e.id FROM {$this->database}.header h, {$this->database}.entry e".
            " WHERE e.id=".$id.
            " AND h.id=e.hdr_id" );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 ) {

            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );

            $exper_id = $attr['exper_id'];
            $experiment = $this->find_experiment_by_id( $exper_id );
            if( is_null( $experiment ))
                throw new LogBookException(
                    __METHOD__,
                    "no experiment for id: {$exper_id} in database. Database can be corrupted." );

            $entry_id = $attr['id'];
            $entry = $experiment->find_entry_by_id( $entry_id );
            if( is_null( $entry ))
                throw new LogBookException(
                    __METHOD__,
                    "no free-form entry for id: {$entry_id} in database. Database can be corrupted." );

            return $entry;
        }
        return null;
    }

    public function delete_entry_by_id( $id ) {
        $entry = $this->find_entry_by_id( $id );
        if( is_null( $entry ))
            throw new LogBookException(
                __METHOD__,
                "no entry for id: {$id} in database. Database can be corrupted." );

        if( is_null( $entry->parent_entry_id()))
            $this->query (
                "DELETE FROM {$this->database}.header WHERE id=(SELECT hdr_id FROM {$this->database}.entry WHERE id={$id})" );
        else
            $this->query (
                "DELETE FROM {$this->database}.entry WHERE id={$id}" );
    }

    public function dettach_entry_from_run( $entry ) {
        $this->query (
            "UPDATE {$this->database}.header SET run_id = NULL WHERE id={$entry->hdr_id()} AND exper_id={$entry->parent()->id()}"
        );
        return $this->find_entry_by_id( $entry->id());
    }
    public function attach_entry_to_run( $entry, $run ) {
        $this->query (
            "UPDATE {$this->database}.header SET run_id = {$run->id()} WHERE id={$entry->hdr_id()} AND exper_id={$entry->parent()->id()}"
        );
        return $this->find_entry_by_id( $entry->id());
    }

    public function find_shift_by_id( $id ) {

        // Find an experiment the shift belongs to
        //
        $result = $this->query (
            "SELECT exper_id FROM {$this->database}.shift WHERE id={$id}" );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 ) {

            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );

            $exper_id = $attr['exper_id'];
            $experiment = $this->find_experiment_by_id( $exper_id );
            if( is_null( $experiment ))
                throw new LogBookException(
                    __METHOD__,
                    "no experiment for id: {$exper_id} in database. Database can be corrupted." );

            $shift = $experiment->find_shift_by_id( $id );
            if( is_null( $shift ))
                throw new LogBookException(
                    __METHOD__,
                    "internal error while looking for shift id: {$id}. Database can be corrupted." );

            return $shift;
        }
        return null;
    }

    public function find_run_by_id( $id ) {

        // Find an experiment the run belongs to
        //
        $result = $this->query (
            "SELECT exper_id FROM {$this->database}.run WHERE id={$id}" );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 ) {

            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );

            $exper_id = $attr['exper_id'];
            $experiment = $this->find_experiment_by_id( $exper_id );
            if( is_null( $experiment ))
                throw new LogBookException(
                    __METHOD__,
                    "no experiment for id: {$exper_id} in database. Database can be corrupted." );

            $run = $experiment->find_run_by_id( $id );
            if( is_null( $run ))
                throw new LogBookException(
                    __METHOD__,
                    "internal error while looking for run id: {$id}. Database can be corrupted." );

            return $run;
        }
        return null;
    }
}
?>
