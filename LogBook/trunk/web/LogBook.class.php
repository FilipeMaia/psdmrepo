<?php

require_once( 'RegDB/RegDB.inc.php' );

class LogBook {

    /* Data members
     */
    private $connection;
    private $regdb;

    /* Constructor
     *
     * Construct the top-level API object using the specified connection
     * parameters. Put null to envorce default values of parameters.
     */
    public function __construct (
        $host     = null,
        $user     = null,
        $password = null,
        $database = null ) {

        $this->connection =
            new LogBookConnection (
                is_null($host)     ? LOGBOOK_DEFAULT_HOST : $host,
                is_null($user)     ? LOGBOOK_DEFAULT_USER : $user,
                is_null($password) ? LOGBOOK_DEFAULT_PASSWORD : $password,
                is_null($database) ? LOGBOOK_DEFAULT_DATABASE : $database);

        /* TODO: Think about a convenient configuration scheme allowing
         * passing connection parameters to the Registration Database
         * in the same fashion it's done for the LogBook connection.
         */
        $this->regdb = new RegDB();
    }

    /*
     * ==========================
     *   TRANSACTION MANAGEMENT
     * ==========================
     */
    public function begin () {
        $this->connection->begin (); }

    public function commit () {
        $this->connection->commit (); }

    public function rollback () {
        $this->connection->rollback (); }

    /* ===============
     *   INSTRUMENTS
     * ===============
     */
    public function instruments () {
        $this->regdb->begin();
        return $this->regdb->instruments();
    }

    /* ===============
     *   EXPERIMENTS
     * ===============
     */
    public function experiments ( $condition = '' ) {

        $list = array();

        $this->regdb->begin();
        $regdb_experiments = $this->regdb->experiments();
        foreach( $regdb_experiments as $e )
            array_push(
                $list,
                new LogBookExperiment (
                    $this->connection,
                    $e ));

        return $list;
    }
    public function experiments_for_instrument ( $name ) {

        $list = array();

        $this->regdb->begin();
        $regdb_experiments = $this->regdb->experiments_for_instrument( $name );
        foreach( $regdb_experiments as $e )
            array_push(
                $list,
                new LogBookExperiment (
                    $this->connection,
                    $e ));

        return $list;
    }

    public function find_experiment_by_id ( $id ) {
        $this->regdb->begin();
        $e = $this->regdb->find_experiment_by_id( $id ) ;
        return is_null( $e ) ?
            null : new LogBookExperiment ( $this->connection, $e ) ;
    }

    public function find_experiment_by_name ( $name ) {
        $this->regdb->begin();
        $e = $this->regdb->find_experiment_by_name( $name ) ;
        return is_null( $e ) ?
            null : new LogBookExperiment ( $this->connection, $e ) ;
    }

    /* ============================
     *   ID-BASED DIRECT LOCATORS
     * ============================
     */
    public function find_attachment_by_id( $id ) {

        $result = $this->connection->query (
            "SELECT h.exper_id, a.entry_id FROM {$this->connection->database}.header h, {$this->connection->database}.entry e, {$this->connection->database}.attachment a".
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
        $result = $this->connection->query (
            "SELECT h.exper_id, e.id FROM {$this->connection->database}.header h, {$this->connection->database}.entry e".
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
            $this->connection->query (
                "DELETE FROM {$this->connection->database}.header WHERE id=(SELECT hdr_id FROM {$this->connection->database}.entry WHERE id={$id})" );
        else
            $this->connection->query (
                "DELETE FROM {$this->connection->database}.entry WHERE id={$id}" );
    }

    public function find_shift_by_id( $id ) {

        // Find an experiment the shift belongs to
        //
        $result = $this->connection->query (
            "SELECT exper_id FROM {$this->connection->database}.shift WHERE id={$id}" );

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
        $result = $this->connection->query (
            "SELECT exper_id FROM {$this->connection->database}.run WHERE id={$id}" );

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
