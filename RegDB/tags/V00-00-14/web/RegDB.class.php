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
        $host      = null,
        $user      = null,
        $password  = null,
        $database  = null,
        $ldap_host = null ) {

        $this->connection =
            new RegDBConnection (
                is_null($host)      ? REGDB_DEFAULT_HOST : $host,
                is_null($user)      ? REGDB_DEFAULT_USER : $user,
                is_null($password)  ? REGDB_DEFAULT_PASSWORD : $password,
                is_null($database)  ? REGDB_DEFAULT_DATABASE : $database,
                is_null($ldap_host) ? REGDB_DEFAULT_LDAP_HOST : $ldap_host );
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

    public function experiments () {
        return $this->find_experiments_by_(); }

    public function experiments_for_instrument ( $instrument_name ) {

        $instrument = $this->find_instrument_by_name( $instrument_name );
        if( !$instrument )
            throw new RegDBException (
                __METHOD__,
                "no such instrument in the database" );

        return $this->find_experiments_by_ ( 'instr_id='.$instrument->id() );
    }

    public function find_experiment ( $instrument_name, $experiment_name ) {

        $instrument = $this->find_instrument_by_name( $instrument_name );
        if( !$instrument )
            throw new RegDBException (
                __METHOD__,
                "no such instrument in the database" );

        $experiments = $this->find_experiments_by_ ( 'instr_id='.$instrument->id()." AND name='{$experiment_name}'" );
        if( count( $experiments ) == 0 ) return null;
        if( count( $experiments ) == 1 ) return $experiments[0];
        throw new RegDBException (
            __METHOD__,
            "inconsistent results returned fromthe database. The database may be corrupted." );
    }

    private function find_experiments_by_ ( $condition='' ) {

        $list = array();

        $extra_condition = $condition == '' ? '' : ' WHERE '.$condition;
        $result = $this->connection->query (
            'SELECT * FROM experiment '.$extra_condition. ' ORDER by begin_time DESC' );

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
        $experiment_name, $instrument_name, $description,
        $registration_time, $begin_time, $end_time,
        $posix_gid, $leader, $contact ) {

        /* Verify parameters
         */
        if( !is_null( $end_time ) && !$begin_time->less( $end_time ))
            throw new RegDBException (
                __METHOD__,
                "begin time '".$begin_time."' isn't less than end time '".$end_time."'" );

        $trim_experiment_name = trim( $experiment_name );
        if( $trim_experiment_name == '' )
            throw new RegDBException (
                __METHOD__,
                "the experiment name can't be empty string" );

        /* Find instrument.
         */
        $instrument = $this->find_instrument_by_name( $instrument_name );
        if( !$instrument )
            throw new RegDBException (
                __METHOD__,
                "no such instrument" );

        /* Make sure the group exists and the group leader is among its
         * members.
         */
        $trim_leader = trim( $leader );
        if( $trim_leader == '' )
            throw new RegDBException (
                __METHOD__,
                "the leader name can't be empty string" );

        $trim_posix_gid = trim( $posix_gid );
        if( $trim_posix_gid == '' )
            throw new RegDBException (
                __METHOD__,
                "the POSIX group name can't be empty string" );
/*
 * TODO: Disable this temporarily untill a more sophisticated algorithm for
 * browsing LDAP associations (groups/users) is implemented.
 *
        if( !$this->is_member_of_posix_group ( $trim_posix_gid, $trim_leader ))
            throw new RegDBException (
                __METHOD__,
                "the proposed leader isn't a member of the POSIX group" );
*/
         /* Proceed with the operation.
          */
        $this->connection->query (
            "INSERT INTO experiment VALUES(NULL,'".$this->connection->escape_string( $trim_experiment_name ).
            "','".$this->connection->escape_string( $description ).
            "',".$instrument->id().
            ",".$registration_time->to64().
            ",".$begin_time->to64().
            ",".$end_time->to64().
            ",'".$this->connection->escape_string( $trim_leader ).
            "','".$this->connection->escape_string( trim( $contact )).
            "','".$this->connection->escape_string( $trim_posix_gid )."')" );

        $experiment = $this->find_experiment_by_id( '(SELECT LAST_INSERT_ID())' );
        if( !$experiment )
            throw new RegDBException (
                __METHOD__,
                "fatal internal error" );

        /* Create the run numbers generator
         */
        $this->connection->query (
            "CREATE TABLE run_{$experiment->id()} ".
            "(num int(11) NOT NULL auto_increment, ".
            " request_time bigint(20) unsigned NOT NULL,".
            " PRIMARY KEY (num)".
            ")" );

        return $experiment;
    }

    public function delete_experiment_by_id ( $id ) {

        $experiment = $this->find_experiment_by_id( $id );
        if( !$experiment )
            throw new RegDBException (
                __METHOD__,
                "no such experiment" );

        /* Proceed with the operation.
         */
        $this->connection->query ( "DELETE FROM experiment_param WHERE exper_id=".$id );
        $this->connection->query ( "DELETE FROM experiment WHERE id=".$id );
        $this->connection->query ( "DROP TABLE IF EXISTS run_".$id );
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

    public function register_instrument ( $instrument_name, $description ) {

        /* Verify parameters
         */
        $trim_instrument_name = trim( $instrument_name );
        if( $trim_instrument_name == '' )
            throw new RegDBException (
                __METHOD__,
                "the instrument name can't be empty string" );

         /* Proceed with the operation.
          */
        $this->connection->query (
            "INSERT INTO instrument VALUES(NULL,'".$this->connection->escape_string( $trim_instrument_name ).
            "','".$this->connection->escape_string( $description )."')" );

        $instrument = $this->find_instrument_by_id( '(SELECT LAST_INSERT_ID())' );
        if( !$instrument )
            throw new RegDBException (
                __METHOD__,
                "fatal internal error" );
        return $instrument;
    }

    public function delete_instrument_by_id ( $id ) {

        $instrument = $this->find_instrument_by_id( $id );
        if( !$instrument )
            throw new RegDBException (
                __METHOD__,
                "no such instrument" );

        /* Find and delete the connected experiments first.
         */
        $experiments = $this->experiments_for_instrument( $instrument->name());
        foreach( $experiments as $e )
            $this->delete_experiment_by_id( $e->id());

        /* Proceed to the instrument.
         */
        $this->connection->query ( "DELETE FROM instrument_param WHERE instr_id=".$id );
        $this->connection->query ( "DELETE FROM instrument WHERE id=".$id );
    }

    /* ====================
     *   GROUPS AND USERS
     * ====================
     */
    public function posix_groups () {
        return $this->connection->posix_groups(); }

    public function is_known_posix_group ( $name ) {
        return $this->connection->is_known_posix_group( $name ); }

    public function is_member_of_posix_group ( $group, $uid ) {
        return $this->connection->is_member_of_posix_group( $group, $uid ); }

    public function posix_group_members ( $name ) {
        return $this->connection->posix_group_members( $name ); }

    public function user_accounts ( $user='*' ) {
        return $this->connection->user_accounts( $user ); }

}

/* =======================
 * UNIT TEST FOR THE CLASS
 * =======================
 *

require_once( "RegDB.inc.php");

try {
    $conn = new RegDB();
    $conn->begin();

    $name = "Exp-A";
    $instrument_name = "CXI";
    $description = "Experiment description goes here";
    $registration_time = LusiTime::now();
    $begin_time = LusiTime::parse( "2009-07-01 09:00:01-0700" );
    $end_time = LusiTime::parse( "2009-09-01 09:00:01-0700" );
    $posix_gid = "lab-users";
    $leader = "perazzo";
    $contact = "Phone: SLAC x5095, E-Mail: gapon@slac.stanford.edu";

    $experiment = $conn->register_experiment (
        $name, $instrument_name, $description,
        $registration_time, $begin_time, $end_time,
        $posix_gid, $leader, $contact );

    print_r( $experiment );

    $conn->commit();

} catch ( RegDBException $e ) {
    print( $e->toHtml());
}
*/
?>
