<?php
class RegDB {

    /* Data members
     */
    private $connection;

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
            new RegDBConnection (
                is_null($host)     ? REGDB_DEFAULT_HOST : $host,
                is_null($user)     ? REGDB_DEFAULT_USER : $user,
                is_null($password) ? REGDB_DEFAULT_PASSWORD : $password,
                is_null($database) ? REGDB_DEFAULT_DATABASE : $database);
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
     *   EXPERIMENTS
     * ===============
     */
    public function experiment_names () {

        $list = array();

        $result = $this->connection->query (
            'SELECT name FROM experiment ' );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push(
                $list,
                mysql_result( $result, $i ));

        return $list;
    }

    public function experiments ( $condition='' ) {

        $list = array();

        $result = $this->connection->query (
            'SELECT * FROM experiment '.$condition );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push(
                $list,
                new RegDBExperiment (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }

    public function find_experiment_by_id ( $id ) {
        return $this->find_experiment_by_( 'id='.$id) ; }

    public function find_experiment_by_name ( $name ) {
        return $this->find_experiment_by_( "name='".$name."'") ; }

    public function find_last_experiment () {
        return $this->find_experiment_by_( 'begin_time=(SELECT MAX(begin_time) FROM experiment)') ; }

    private function find_experiment_by_ ( $condition ) {

        $result = $this->connection->query(
            'SELECT * FROM "experiment" WHERE '.$condition );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new RegDBExperiment(
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        return null;
    }

    public function register_experiment (
        $name, $instrument, $description,
        $registration_time, $begin_time, $end_time,
        $posix_gid, $leader, $contact ) {

        /* Verify parameters
         */
        if( !is_null( $end_time ) && !$begin_time->less( $end_time ))
            throw new RegDBException(
                __METHOD__,
                "begin time '".$begin_time."' isn't less than end time '".$end_time."'" );

        /* Find instrument.
         */
        ;

        $this->connection->query (
            "INSERT INTO experiment VALUES(NULL,'".$name
            ."',".$begin_time->to64()
            .",".$end_time->to64().")" );

        return $this->find_experiment_by_id( '(SELECT LAST_INSERT_ID())' );
    }

    /* ===============
     *   INSTRUMENTS
     * ===============
     */
    public function instrument_names () {

        $list = array();

        $result = $this->connection->query (
            'SELECT name FROM instrument ' );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push(
                $list,
                mysql_result( $result, $i ));

        return $list;
    }

    public function instruments ( $condition='' ) {

        $list = array();

        $result = $this->connection->query (
            'SELECT * FROM instrument '.$condition );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push(
                $list,
                new RegDBInstrument (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }

    public function find_instrument_by_id ( $id ) {
        return $this->find_instrument_by_( 'id='.$id) ; }

    public function find_instrument_by_name ( $name ) {
        return $this->find_instrument_by_( "name='".$name."'") ; }

    private function find_instrument_by_ ( $condition ) {

        $result = $this->connection->query(
            'SELECT * FROM instrument WHERE '.$condition );

        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows == 1 )
            return new RegDBInstrument (
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        throw new RegDBException(
            __METHOD__,
            "unexpected size of result set returned by the query" );
    }
 }
?>
