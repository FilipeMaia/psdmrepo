<?php
/**
 * Class RegDBExperiment an abstraction for experiments.
 *
 * @author gapon
 */
class RegDBExperiment {

    /* Data members
     */
    private $connection;
    private $registry;

    public $attr;

    /* Constructor
     */
    public function __construct ( $connection, $registry, $attr ) {
        $this->connection = $connection;
        $this->registry = $registry;
        $this->attr = $attr;
    }

    public function parent () {
        return $this->registry; }

    public function id () {
        return $this->attr['id']; }

    public function name () {
        return $this->attr['name']; }

    public function description () {
        return $this->attr['descr']; }

    public function instr_id () {
        return $this->attr['instr_id']; }

    public function instrument () {
        $result = $this->parent()->find_instrument_by_id( $this->instr_id());
        if( is_null( $result ))
            throw new RegDBException (
                __METHOD__,
                "no instrument found - database may be in the inconsistent state" );
        return $result;
    }

    public function registration_time () {
        return LusiTime::from64( $this->attr['registration_time'] ); }

    public function begin_time () {
        return LusiTime::from64( $this->attr['begin_time'] ); }

    public function end_time () {
        if( is_null( $this->attr['end_time'] )) return null;
        return LusiTime::from64( $this->attr['end_time'] ); }

    public function leader_account () {
        return $this->attr['leader_account']; }

    public function contact_info () {
        return $this->attr['contact_info']; }

    public function POSIX_gid () {
        return $this->attr['posix_gid']; }

    public function group_member_accounts () {
        $result = array();
        $members = $this->connection->posix_group_members( $this->POSIX_gid());
        foreach( $members as $member )
            array_push( $result, $member['uid'] );
        return $result;
    }

    public function group_members () {
        return $this->connection->posix_group_members( $this->POSIX_gid()); }

    public function in_interval ( $timestamp ) {
        return LusiTime::in_interval(
            $timestamp,
            $this->attr['begin_time'],
            $this->attr['end_time'] ); }

    /* =============
     *   MODIFIERS
     * =============
     */
    public function set_description ( $description ) {
        $result = $this->connection->query(
            "UPDATE experiment SET descr='".$this->connection->escape_string( $description ).
            "' WHERE id=".$this->id());
        $this->attr['descr'] = $description;
    }

    public function set_contact_info ( $contact_info ) {
        $result = $this->connection->query(
            "UPDATE experiment SET contact_info='".$this->connection->escape_string( $contact_info ).
            "' WHERE id=".$this->id());
        $this->attr['contact_info'] = $contact_info;
    }

    public function set_interval( $begin_time, $end_time ) {

        /* Verify parameters
         */
        if( !is_null( $end_time ) && !$begin_time->less( $end_time ))
            throw new RegDBException (
                __METHOD__,
                "begin time '".$begin_time."' isn't less than end time '".$end_time."'" );

        $begin_time_64 = $begin_time->to64();
        $end_time_64 = $end_time->to64();

        /* Proceed with the operation.
         */
        $this->connection->query (
            "UPDATE experiment SET begin_time=".$begin_time_64.
            ", end_time=".$end_time_64.
            " WHERE id=".$this->id());

        $this->attr['begin_time'] = $begin_time_64;
        $this->attr['end_time'] = $end_time_64;
    }

    /* ==============
     *   PARAMETERS
     * ==============
     */
    public function num_params ( $condition='' ) {

        $extra_condition = $condition == '' ? '' : ' AND '.$condition;
        $result = $this->connection->query(
            'SELECT COUNT(*) FROM experiment_param WHERE exper_id='.$this->id().$extra_condition );

        if( $nrows == 1 )
            return mysql_result( $result, 0 );

        throw new RegDBException (
            __METHOD__,
            "unexpected size of result set returned by the query" );
    }

    public function param_names ( $condition='' ) {

        $list = array();

        $extra_condition = $condition == '' ? '' : ' AND '.$condition;
        $result = $this->connection->query(
            'SELECT param FROM experiment_param WHERE exper_id='.$this->id().$extra_condition );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push(
                $list,
                mysql_result( $result, $i ));

        return $list;
    }

    public function params ( $condition='' ) {

        $list = array();

        $extra_condition = $condition == '' ? '' : ' AND '.$condition;
        $result = $this->connection->query(
            'SELECT * FROM experiment_param WHERE exper_id='.$this->id().$extra_condition );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new RegDBExperimentParam (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }

    public function find_param_by_name ( $name ) {
        return $this->find_param_by_( "param='".$this->connection->escape_string( trim( $name ))."'" ); }

    private function find_param_by_ ( $condition ) {

        $extra_condition = $condition == '' ? '' : ' AND '.$condition;
        $result = $this->connection->query(
            'SELECT * FROM experiment_param WHERE exper_id='.$this->id().$extra_condition );

        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows == 1 )
            return new RegDBExperimentParam (
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        throw new RegDBException(
            __METHOD__,
            "unexpected size of result set returned by the query" );
    }

    public function add_param ( $name, $value, $description ) {

        /* Verify the leader's name
         */
        if( is_null( $name ) || is_null( $value ) || is_null( $description ))
            throw new RegDBException (
                __METHOD__, "method parameters can't be null objects" );

        $trimmed_name = trim( $name );
        if( strlen( $trimmed_name ) == 0 )
            throw new RegDBException(
                __METHOD__, "parameter name can't be empty" );

        /* Proceed with the operation.
         */
        $this->connection->query (
            'INSERT INTO experiment_param VALUES('.$this->id().
            ",'".$this->connection->escape_string( $trimmed_name ).
            "','".$this->connection->escape_string( $value ).
            "','".$this->connection->escape_string( $description )."')" );

        $new_param = $this->find_param_by_( "param='".$this->connection->escape_string( $trimmed_name )."'" );
        if( is_null( $new_param ))
            throw new RegDBException (
                __METHOD__,
                "internal implementation errort" );

        return $new_param;
    }

    public function remove_param ( $name ) {

        /* Verify values of parameters
         */
        if( is_null( $name ))
            throw new RegDBException (
                __METHOD__, "method parameters can't be null objects" );

        $trimmed_name = trim( $name );
        if( strlen( $trimmed_name ) == 0 )
            throw new RegDBException(
                __METHOD__, "parameter name can't be empty" );

        /* Make sure the parameter is known.
         */
        $param = $this->find_param_by_name ( $name );
        if( is_null( $param ))
            throw new RegDBException(
                __METHOD__, "parameter doesn't exist in the database" );

        /* Proceed with the operation.
         */
        $this->connection->query (
            "DELETE FROM experiment_param WHERE param='{$trimmed_name}' AND exper_id={$this->id()})" );
    }

    public function remove_all_params () {

        /* Proceed with the operation.
         */
        $this->connection->query (
            "DELETE FROM experiment_param WHERE exper_id={$this->id()}" );
    }

    /* ===============
     *   RUN NUMBERS
     * ===============
     */
    public function runs () {

        $list = array();
        $table = "run_".$this->id();

        $result = $this->connection->query(
            "SELECT * FROM {$table} ORDER BY num" );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new RegDBRun (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));

        return $list;
    }

    public function generate_run () {

        $table = "run_".$this->id();
        $request_time = LusiTime::now()->to64();

        $this->connection->query (
            "INSERT INTO {$table} VALUES(NULL,{$request_time})" );

        $result = $this->connection->query(
            "SELECT * FROM {$table} WHERE num=(SELECT LAST_INSERT_ID())" );

        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows == 1 )
            return new RegDBRun (
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        throw new RegDBException(
            __METHOD__,
            "unexpected size of result set returned by the query" );
    }

    public function last_run () {

        $list = array();
        $table = "run_".$this->id();

        $result = $this->connection->query(
            "SELECT * FROM {$table} WHERE num=(SELECT MAX(num) FROM {$table})" );

        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows == 1 )
            return new RegDBRun (
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        throw new RegDBException(
            __METHOD__,
            "unexpected size of result set returned by the query" );

    }
}
?>
