<?php
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

    public function id () {
        return $this->attr['id']; }

    public function name () {
        return $this->attr['name']; }

    public function description () {
        return $this->attr['descr']; }

    /* ==============
     *   PARAMETERS
     * ==============
     */
    public function num_params ( $condition='' ) {

        $extra_condition = $condition == '' ? '' : ' AND '.$condition;
        $result = $this->connection->query(
            'SELECT COUNT(*) FROM instrument_param WHERE instr_id='.$this->id().$extra_condition );

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
            'SELECT param FROM instrument_param WHERE instr_id='.$this->id().$extra_condition );

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
            'SELECT * FROM instrument_param WHERE instr_id='.$this->id().$extra_condition );

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
        return $this->find_param_by_( "param='".mysql_real_escape_string( trim( $name ))."'" ); }

    private function find_param_by_ ( $condition ) {

        $extra_condition = $condition == '' ? '' : ' AND '.$condition;
        $result = $this->connection->query(
            'SELECT * FROM instrument_param WHERE instr_id='.$this->id().$extra_condition );

        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows == 1 )
            return new RegDBInstrumentParam (
                $this->connection,
                this,
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
            'INSERT INTO instrument_param VALUES('.$this->id().
            ",'".mysql_real_escape_string( $trimmed_name ).
            "','".mysql_real_escape_string( $value ).
            ",'".mysql_real_escape_string( $description )."')" );

        $new_param = $this->find_param_by_( "param='".mysql_real_escape_string( $trimmed_name )."'" );
        if( is_null( $new_param ))
            throw new RegDBException (
                __METHOD__,
                "internal implementation errort" );

        return $new_param;
    }
}
?>
