<?php

namespace RegDB;

require_once( 'regdb.inc.php' );

/**
 * Class RegDBInstrument an abstraction for instruments.
 *
 * @author gapon
 */
class RegDBInstrument {

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

    public function regdb () {
        return $this->registry; }

    public function id () {
        return $this->attr['id']; }

    public function name () {
        return $this->attr['name']; }

    public function description () {
        return $this->attr['descr']; }

    public function is_standard () {
        $is_standard_param = $this->find_param_by_name ('isStandard');
    	return !is_null( $is_standard_param ) && ( $is_standard_param->value() != '0' );
    }

    public function is_location () {
        $is_location_param = $this->find_param_by_name ('isLocation');
    	return !is_null( $is_location_param ) && ( $is_location_param->value() != '0' );
    }

    public function is_mobile () {
        $is_mobile_param = $this->find_param_by_name ('isMobile');
    	return !is_null( $is_mobile_param ) && ( $is_mobile_param->value() != '0' );
    }

    public function experiments () {
        return $this->parent()->experiments_for_instrument($this->name());
    }

    /* =============
     *   MODIFIERS
     * =============
     */
    public function set_description ( $description ) {
        $result = $this->connection->query(
            "UPDATE {$this->connection->database}.instrument SET descr='".$this->connection->escape_string( $description ).
            "' WHERE id=".$this->id());
        $this->attr['descr'] = $description;
    }

    /* ==============
     *   PARAMETERS
     * ==============
     */
    public function num_params ( $condition='' ) {

        $extra_condition = $condition == '' ? '' : ' AND '.$condition;
        $result = $this->connection->query(
            "SELECT COUNT(*) FROM {$this->connection->database}.instrument_param WHERE instr_id=".$this->id().$extra_condition );

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
            "SELECT param FROM {$this->connection->database}.instrument_param WHERE instr_id=".$this->id().$extra_condition );

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
            "SELECT * FROM {$this->connection->database}.instrument_param WHERE instr_id=".$this->id().$extra_condition );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ )
            array_push (
                $list,
                new RegDBInstrumentParam (
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
            "SELECT * FROM {$this->connection->database}.instrument_param WHERE instr_id=".$this->id().$extra_condition );

        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows == 1 )
            return new RegDBInstrumentParam (
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        throw new RegDBException(
            __METHOD__,
            "unexpected size of result set returned by the query" );
    }

    public function add_param ( $name, $value, $description ) {

        /* Verify values of parameters
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
            "INSERT INTO {$this->connection->database}.instrument_param VALUES(".$this->id().
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
            "DELETE FROM {$this->connection->database}.instrument_param WHERE param='{$trimmed_name}' AND instr_id={$this->id()})" );
    }

    public function remove_all_params () {

        /* Proceed with the operation.
         */
        $this->connection->query (
            "DELETE FROM {$this->connection->database}.instrument_param WHERE instr_id={$this->id()}" );
    }
}
?>
