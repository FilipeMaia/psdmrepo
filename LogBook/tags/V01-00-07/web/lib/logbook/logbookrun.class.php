<?php

namespace LogBook;

require_once( 'logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LusiTime\LusiTime;

class LogBookRun {

    /* Data members
     */
    private $logbook;
    private $experiment;

    public $attr;

    /* Constructor
     */
    public function __construct ( $logbook, $experiment, $attr ) {
        $this->logbook    = $logbook;
        $this->experiment = $experiment;
        $this->attr       = $attr;
    }

    /* Accessors
     */
    public function parent     () { return                  $this->experiment; }
    public function id         () { return           intval($this->attr['id']); }
    public function num        () { return           intval($this->attr['num']); }
    public function exper_id   () { return           intval($this->attr['exper_id']); }
    public function begin_time () { return LusiTime::from64($this->attr['begin_time']); }
    public function end_time   () { return          is_null($this->attr['end_time']) ? null : LusiTime::from64($this->attr['end_time']); }
 
    public function in_interval( $time ) {
        return LusiTime::in_interval (
            $time,
            $this->begin_time(),
            $this->end_time() );
    }

    /**
     * Return a portable array representation which can be translated into JSON
     *
     * @return array()
     */
    public function to_array () {

        $begin_time = $this->begin_time() ;
        $end_time   = $this->end_time() ;

        return array (
            'exper_id' => $this->exper_id() ,
            'id'       => $this->id() ,
            'num'      => $this->num() ,
            'begin_time'  => array (
                'as_str'  => $begin_time->toStringShort() ,
                'as_sec'  => $begin_time->sec ,
                'as_nsec' => $begin_time->nsec ,
                'as_64'   => $begin_time->to64()
            ) ,
            'end_time'    => array (
                'as_str'  => $end_time ? $end_time->toStringShort() : '' ,
                'as_sec'  => $end_time ? $end_time->sec             : 0 ,
                'as_nsec' => $end_time ? $end_time->nsec            : 0 ,
                'as_64'   => $end_time ? $end_time->to64()          : 0
            )
        ) ;
    }

    /*
     * =============================
     *   SUMMARY PARAMETERS VALUES
     * =============================
     */
    public function values ( $condition='', $return_dict=false ) {

        $list = array();
        $run_id = $this->id();

        /* Get descriptions of run parameters for the experiment. We need to know
         * type names of the parameters to scedule a request to the corresponding
         * tables.
         */
        $params = $this->experiment->run_params();
        foreach( $params as $p ) {

            $param_id = $p->id();
            $param    = $p->name();
            $type     = $p->type_name();

            $extra_condition = $condition == '' ? '' : ' AND '.$condition;
            $result = $this->logbook->query (
                "SELECT p.*,v.val FROM {$this->logbook->database}.run_val p, {$this->logbook->database}.run_val_".$type.' v WHERE p.run_id='.$run_id.
                ' AND p.param_id='.$param_id.
                ' AND p.run_id=v.run_id AND p.param_id=v.param_id'.
                $extra_condition );

            $nrows = mysql_numrows( $result );
            for( $i = 0; $i < $nrows; $i++ ) {
            	if( $return_dict )
            	    $list[$param] =
                        new LogBookRunVal (
                            $this->logbook,
                            $this,
                            $param,
                            $type,
                            mysql_fetch_array( $result, MYSQL_ASSOC ));
            	else
                    array_push(
                        $list,
                        new LogBookRunVal (
                            $this->logbook,
                            $this,
                            $param,
                            $type,
                            mysql_fetch_array( $result, MYSQL_ASSOC )));
            }
        }
        return $list;
    }

    /* Get a value of the specified run parameter
     */
    public function get_param_value ( $param ) {

        /* Find the parameter and get its identifier and its type.
         * Also prepare the value for the specified SQL type.
         */
        $param = $this->experiment->find_run_param_by_name( $param );
        if( is_null( $param ))
            throw new LogBookException (
                __METHOD__,
                "no such run parameter: '".$param."'" );

        $param_id    = $param->attr['id'];
        $type        = $param->attr['type'];
        $value_table = 'run_val_'.$type;

        /* Fetch the value and the bookkeeping info
         */
        $result = $this->logbook->query (
            "SELECT p.*,v.val FROM {$this->logbook->database}.run_val p, ".$value_table.' v WHERE p.run_id='.$this->id().
            ' AND p.param_id='.$param_id.
            ' AND p.run_id=v.run_id AND p.param_id=v.param_id' );

        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows != 1 )
            throw new LogBookException (
                __METHOD__,
                "unexpected size of the result set returned by query" );

        return new LogBookRunVal (
            $this->logbook,
            $this,
            $param,
            $type,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }

    /* Set a value of the specified run parameter
     */
    public function set_param_value ( $param, $value, $source, $updated, $allow_update=false ) {

        /* Find the parameter and get its identifier and its type.
         * Also prepare the value for the specified SQL type.
         */
        $param = $this->experiment->find_run_param_by_name( $param );
        if( is_null( $param ))
            throw new LogBookException (
                __METHOD__,
                "no such run parameter: '".$param."'" );

        $param_id = $param->attr['id'];
        $type     = $param->attr['type'];

        if( $type == 'TEXT') {
            $value4sql = "'".$value."'";

        } else if( $type =='INT' ) {
            if( 1 != sscanf( $value, "%d", $value4sql ))
                throw new LogBookException (
                    __METHOD__,
                    "not an integer value of the parameter: ".$value );

        } else if( $type =='DOUBLE' ) {
            if( 1 != sscanf( $value, "%lf", $value4sql ))
                throw new LogBookException (
                    __METHOD__,
                    "not a double precision value of the parameter: ".$value );

        } else {
            /* Treat it as the string */
            $value4sql = "'".$value."'";
        }
        $value_table = "{$this->logbook->database}.run_val_".$type;

        /* Check if its value is already set, and if so - if we're allowed
         * to update it.
         */
        $run_id = $this->id();

        $result = $this->logbook->query (
            "SELECT COUNT(*) AS \"count\" FROM {$this->logbook->database}.run_val p WHERE p.run_id=".$run_id.
            ' AND p.param_id='.$param_id );

        $nrows = mysql_numrows( $result );
        if( $nrows != 1 )
            throw new LogBookException (
                __METHOD__,
                "unexpected size of the result set returned by query" );

        $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
        if( $attr['count'] > 0 ) {

            if( !$allow_update )
                throw new LogBookException (
                    __METHOD__,
                    "the value of parameter: '".$param."' was set before and it's not allowed to be updated" );

            $this->logbook->query (
                "UPDATE {$this->logbook->database}.run_val SET source='".$source."', updated=".$updated->to64().
                ' WHERE run_id='.$run_id.' AND param_id='.$param_id );

            $this->logbook->query(
                "UPDATE ".$value_table." SET val=".$value4sql.
                ' WHERE run_id='.$run_id.' AND param_id='.$param_id );

        } else {
            $this->logbook->query (
                "INSERT INTO {$this->logbook->database}.run_val VALUES (".$run_id.",".$param_id.",'".$source."',".$updated->to64().")" );

            $this->logbook->query (
                "INSERT INTO ".$value_table." VALUES (".$run_id.",".$param_id.",".$value4sql.")" );
        }

        /* Fetch the value and the bookkeeping info
         */
        $result = $this->logbook->query (
            "SELECT p.*,v.val FROM {$this->logbook->database}.run_val p, ".$value_table.' v WHERE p.run_id='.$run_id.
            ' AND p.param_id='.$param_id.
            ' AND p.run_id=v.run_id AND p.param_id=v.param_id'.
            $extra_condition );

        $nrows = mysql_numrows( $result );
        if( $nrows != 1 )
            throw new LogBookException (
                __METHOD__,
                "unexpected size of the result set returned by query" );

        return new LogBookRunVal (
            $this->logbook,
            $this,
            mysql_fetch_array( $result, MYSQL_ASSOC ));
    }

    /*
     * ======================
     *   PER-RUN ATTRIBUTES
     * ======================
     */
    public function attr_classes() {
        $result = $this->logbook->query("SELECT DISTINCT class FROM {$this->logbook->database}.run_attr WHERE run_id={$this->id()} ORDER BY class");
        $list = array();
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            array_push($list, $attr['class']);
        }
        return $list;
    }
    public function attributes($class_name) {
        $class_name_sql = is_null($class_name) ? '' : $this->logbook->escape_string($class_name);
        if($class_name_sql == '')
            throw new LogBookException (
                __METHOD__,
                "no attribute class name provided");
        $result = $this->logbook->query("SELECT * FROM {$this->logbook->database}.run_attr WHERE run_id={$this->id()}  AND class='{$class_name_sql}' ORDER BY name");
        $list = array();
        for( $i = 0, $nrows = mysql_numrows( $result ); $i < $nrows; $i++ ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            array_push(
                $list,
                new LogBookRunAttr (
                    $this->logbook,
                    $this,
                    $attr,
                    $this->attr_val($attr['id'],$attr['type'])));
        }
        return $list;
    }
    private function attr_val($attr_id, $attr_type) {
        $result = $this->logbook->query("SELECT val FROM {$this->logbook->database}.run_attr_{$attr_type} WHERE attr_id={$attr_id}");
        $nrows = mysql_numrows( $result );
        if( $nrows != 1 )
            throw new LogBookException (
                __METHOD__,
                "no attribute value found for attr_id: {$attr_id} and type: {$attr_type}; the database may be corrupt");
        $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
        return $attr['val'];
    }
    private function attr_id ($type, $class, $name) {
        $class_escaped = $this->logbook->escape_string($class) ;
        $name_escaped  = $this->logbook->escape_string($name) ;
        $result = $this->logbook->query("SELECT id FROM {$this->logbook->database}.run_attr WHERE run_id={$this->id()} AND class='{$class_escaped}' AND name='{$name_escaped}' AND type='{$type}'");
        if (mysql_numrows($result) > 0) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            return intval($attr['id']) ;
        }
        return null ;
    }

    public function set_attr_val_INT    ($class, $name, $val=0,  $descr='') { $this->set_attr_val_('INT',    $class, $name, $val, $descr) ; }
    public function set_attr_val_DOUBLE ($class, $name, $val=0., $descr='') { $this->set_attr_val_('DOUBLE', $class, $name, $val, $descr) ; }
    public function set_attr_val_TEXT   ($class, $name, $val='', $descr='') { $this->set_attr_val_('TEXT',   $class, $name, $val, $descr) ; }

    private function set_attr_val_ ($type, $class, $name, $val, $descr) {
        $class_escaped = $this->logbook->escape_string($class) ;
        $name_escaped  = $this->logbook->escape_string($name) ;
        $descr_escaped = $this->logbook->escape_string($descr) ;
        $val_escaped_quoted_if_needed = $type === 'TEXT' ? "'".$this->logbook->escape_string($val)."'" : $val ;
        $id = $this->attr_id($type, $class, $name) ;
        if ($id) {
            $this->logbook->query("UPDATE {$this->logbook->database}.run_attr_{$type} SET val={$val_escaped_quoted_if_needed} WHERE attr_id={$id}") ;
        } else {
            $this->logbook->query("INSERT INTO {$this->logbook->database}.run_attr VALUES(NULL,{$this->id()},'{$class_escaped}','{$name_escaped}','{$type}','{$descr_escaped}')") ;
            $this->logbook->query("INSERT INTO {$this->logbook->database}.run_attr_{$type} VALUES((SELECT LAST_INSERT_ID()),{$val_escaped_quoted_if_needed})") ;
        }
    }

    /* =================
     *   MANAGING RUNS
     * =================
     */

    /* Close the open-ended run
     */
    public function close ( $end_time ) {

        if( !is_null($this->attr['end_time']))
            throw new LogBookException (
                __METHOD__,
                "run '".$this->num()."' is already closed" );

        /* Verify the value of the parameter
         */
        if( is_null( $end_time ))
            throw new LogBookException (
                __METHOD__,
                "end time can't be null" );

        if( !$this->begin_time()->less( $end_time ))
            throw new LogBookException (
                __METHOD__,
                "end time '".$end_time."' isn't greater than the begin time" );

        /* Also make sure the end time of the run doesn't go beyond the begin
         * time of the next run (if there is such run).
         */
        $next_run = $this->parent()->find_next_run_for( $this );
        if( !is_null( $next_run )) {
            print_r( $next_run );
            if( !$this->begin_time()->greaterOrEqual( $end_time ))
                throw new LogBookException( 
                    __METHOD__,
                    "end time '".$end_time."' isn't less or equal to the begin time of the next run" );
        }

        /* Make the update
         */
        $end_time_64 = LusiTime::to64from( $end_time );
        $this->logbook->query (
            "UPDATE {$this->logbook->database}.run SET end_time=".$end_time_64.
            ' WHERE exper_id='.$this->exper_id().' AND num='.$this->attr['num'] );

        /* Update the current state of the object
         */
        $this->attr['end_time'] = $end_time_64;
    }

    /* =====================
     *   FREE-FORM ENTRIES
     * =====================
     */
    public function entries () {
        return $this->parent()->entries_of_run( $this->id());
    }

    /* =========
     *   SHIFT
     * ==========
     */
    public function shift () {
        $my_shift = $this->parent()->find_shift_at( $this->attr['begin_time'] );
        if( is_null( $my_shift ))
            throw new LogBookException (
                __METHOD__,
                "run id:{$this->id()} has no overlapping shift. Database may be corrupted." );
        return $my_shift;
    }
}
?>
