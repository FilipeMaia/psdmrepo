<?php

namespace LogBook ;

require_once 'logbook.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;
require_once 'filemgr/filemgr.inc.php' ;

use LusiTime\LusiTime ;

use FileMgr\DbConnection ;
use FileMgr\FileMgrException ;

class LogBookRunTable {

    /* Data members
     */
    private $experiment ;

    private $attr ;

    /** Constructor
     */
    public function __construct ($experiment, $attr) {
        $this->experiment = $experiment ;
        $this->attr       = $attr ;
    }

    /* Accessors
     */
    public function experiment    () { return                  $this->experiment ; }
    public function id            () { return                  $this->attr['id'] ; }
    public function name          () { return                  $this->attr['name'] ; }
    public function descr         () { return                  $this->attr['descr'] ; }
    public function created_uid   () { return                  $this->attr['created_uid'] ; }
    public function created_time  () { return LusiTime::from64($this->attr['created_time']) ; }
    public function modified_uid  () { return                  $this->attr['modified_uid'] ; }
    public function modified_time () { return LusiTime::from64($this->attr['modified_time']) ; }


    /* SQL helpers
     */
    private function query              ($sql) { return $this->experiment()->logbook()->query($sql) ; }
    private function database           ()     { return $this->experiment()->logbook()->database ; }
    private function trim_escape_string ($str) { return $this->experiment()->logbook()->escape_string(trim($str)) ; }

    /* Operations
     */
    public function columns () {
        $sql = "SELECT * FROM {$this->database()}.run_table_coldef WHERE table_id={$this->id()} ORDER BY position, name" ;
        $result = $this->query($sql) ;
        $list = array() ;
        for ($nrows=mysql_numrows($result), $i=0; $i<$nrows; $i++) {
            $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
            array_push($list, $attr) ;
        }
        return $list ;
    }
    public function find_column_by_id ($id) {
        $sql = "SELECT * FROM {$this->database()}.run_table_coldef WHERE id=$id" ;
        $result = $this->query($sql) ;
        $nrows=mysql_numrows($result) ;
        if (!$nrows) return null ;
        if ($nrows === 1) {
            $attr = mysql_fetch_array($result, MYSQL_ASSOC) ;
            return $attr ;
        }
        throw new LogBookException (
                __METHOD__ ,
                "internal error") ;
    }
    
    public function add_column ($name, $type, $source, $is_editable, $position) {
        $this->query (
            "INSERT INTO {$this->database()}.run_table_coldef VALUES(NULL,".$this->id()
            .",'" .$this->trim_escape_string($name)
            ."','".$this->trim_escape_string($type)
            ."','".$this->trim_escape_string($source)
            ."','".($is_editable ? 'YES' : 'NO')
            ."',".$position
            .")"
        ) ;
    }
    public function remove_column_by_id ($id) {
        $id = intval($id) ;
        $this->query (
            "DELETE FROM {$this->database()}.run_table_coldef WHERE table_id={$this->id()} AND id={$id}"
        ) ;
    }
    public function row ($run, $columns) {
        $row = array() ;
        foreach ($columns as $col) {
            $id = $col['id'] ;
            $name = $col['name'] ;
            $type = $col['type'] ;
            $source = $col['source'] ;

            $row[$name] = '' ;

            switch ($type) {

                case 'Editable':

                    $sql = "SELECT val FROM {$this->database()}.run_table_text WHERE coldef_id={$id} AND run_id={$run->id()}" ;
                    $result = $this->query($sql) ;
                    $nrows = mysql_numrows($result) ;
                    if ($nrows) {
                        $attr = mysql_fetch_array($result, MYSQL_NUM) ;
                        $row[$name] = $attr[0] ;
                    } else {
                        $row[$name] = '' ;
                    }
                    break ;

                case 'Run Info':

                    switch ($source) {
                        case 'Run Title':
                            foreach ($run->attributes('Run Info') as $attr) {
                                if ($attr->name() == $source) {
                                    $row[$name] = $attr->val() ;
                                    break ;
                                }
                            }
                            break ;
                        case 'Run Duration':
                            $row[$name] = $run->end_time() ? LogBookUtils::format_seconds_1($run->end_time()->sec - $run->begin_time()->sec) : '' ;
                            break ;
                    }
                    break ;

                case 'DAQ_Detectors':
                case 'DAQ Detectors':
                    foreach (array('DAQ_Detectors', 'DAQ Detectors') as $class) {
                        foreach ($run->attributes($class) as $attr) {
                            if ($attr->name() === $source) {
                                $row[$name] = 1 ;
                                break ;
                            }
                        }
                    }
                    break ;

                case 'DAQ Detector Totals':
                    foreach ($run->attributes('DAQ Detector Totals') as $attr) {
                        if ($attr->name() === $source) {
                            $row[$name] = $attr->val() ;
                            break ;
                        }
                    }
                    break ;

                case 'Calibrations':
                    foreach ($run->attributes('Calibrations') as $attr) {
                        if ($attr->name() === $source) {
                            $row[$name] = $attr->val() ? 'Y' : ' ' ;
                            break ;
                        }
                    }
                    break ;

                default:
                    
                    if (substr($type, 0, 5) === 'EPICS') {
                        $param = $this->experiment()->find_run_param_by_name($source) ;
                        if ($param) {
                            $param_value = $run->get_param_value($source) ;
                            if ($param_value) $row[$name] = $param_value->value() ;
                        }
                    }
                    break ;
            }
        }
        return $row ;
    }
    
    public function update ($run_id, $coldef_id, $value, $uid) {

        $now   = LusiTime::now()->to64() ;
        $uid   = $this->trim_escape_string($uid) ;
        $value = $this->trim_escape_string($value) ;

        $col = $this->find_column_by_id($coldef_id) ;
        if (!$col)
            throw new LogBookException (
                __METHOD__ ,
                "internal error: illegal column identifier: '{$coldef_id}'") ;

        $coltype   = $col['type'] ;
        $colsource = $col['source'] ;

        $updated = false ;

        switch ($coltype) {

            case 'Run Info':
                switch ($colsource) {
                    case 'Run Title':
                        $run = $this->experiment()->find_run_by_id($run_id) ;
                        if (!$run)
                            throw new LogBookException (
                                __METHOD__ ,
                                "internal error: illegal run identifier: '{$run_id}'") ;
                        $run->set_attr_val_TEXT('Run Info', 'Run Title', $value) ;
                        $updated = true ;
                        break ;
                }
                break ;

            case 'Editable':
                // -- try to insert first. If the row already exists then update the one --
                try {
                    $this->query("INSERT INTO {$this->database()}.run_table_text VALUES({$run_id},{$coldef_id},'{$value}','{$uid}',{$now})") ;
                } catch (FileMgrException $e) {
                    if (is_null($e->errno) || ($e->errno != DbConnection::$ER_DUP_ENTRY)) throw $e;
                    $this->query("UPDATE {$this->database()}.run_table_text SET val='{$value}',modified_uid='{$uid}',modified_time={$now} WHERE coldef_id={$coldef_id} AND run_id={$run_id}") ;
                }
                $updated = true ;
                break ;
        }
        if (!$updated)
            throw new LogBookException (
                __METHOD__ ,
                "internal error: unsupported column type: '{$coltype}' or source: '{$colsource}'") ;

        $this->query("UPDATE {$this->database()}.run_table SET modified_uid='{$uid}',modified_time={$now} WHERE id={$this->id()}") ;
    }
    
    
    public function reconfigure ($name, $descr, $uid, $coldef) {
        
        $now64 = LusiTime::now()->to64() ;
        $uid = $this->trim_escape_string($uid) ;        

        // Update table metadata

        $this->query (
            "UPDATE {$this->database()}.run_table SET".
            " name='".$this->trim_escape_string($name)."',".
            " descr='".$this->trim_escape_string($descr)."',".
            " modified_uid='".$this->trim_escape_string($uid)."',".
            " modified_time={$now64}".
            " WHERE id={$this->id()}"
        ) ;

        // Update/extend column definitions.
        // 
        // Note the following:
        //
        // - new colums are added if their identifiers are found to be equal to 0
        // - changing a type of an existing column from 'Editable' would result in recreating the column definition
        //   because we need to delete the data entries associated with that colum.

        foreach ($coldef as $col) {
            
            // -- New column to be added --

            if (!$col->id) {
                $this->add_column_using_($col) ;
                continue ;
            }
            
            $old_col = $this->find_column_by_id($col->id) ;
            if (!$old_col)
                throw new LogBookException (
                    __METHOD__ ,
                    "internal error: column not found for id: {$col->id}") ;

            // -- Recreate (remove then add) the column if changing its type from 'Editable'
            //    to someting else --

            if (($old_col['type'] === 'Editable') && ($old_col['type'] !== $col->type)) {
                $this->remove_column_by_id($col->id) ;
                $this->add_column_using_($col) ;
                continue ;
            }
            
            // -- Update column definition --

            $is_editable = false ;
            switch ($col->type) {
                case 'Editable':
                    $is_editable = true ;
                    break ;
                case 'Run Info':
                    if ($col->source === 'Run Title') $is_editable = true ;
                    break ;
            }
            $this->query (
                "UPDATE {$this->database()}.run_table_coldef SET".
                " name='".$this->trim_escape_string($col->name)."',".
                " type='".$this->trim_escape_string($col->type)."',".
                " source='".$this->trim_escape_string($col->source)."',".
                " is_editable='".($is_editable ? 'YES' : 'NO')."',".
                " position={$col->position}".
                " WHERE id={$col->id}"
            ) ;
        }
        
        // Remove existing collumns which aren't found among parameters of
        // the method.
        //
        // NOTES:
        // - this step should be run after applying modifications to
        //   existing columns.
        // - a column becomes a subject for the deletion if there is no full
        //   match of its triplet: (name,type,source).

        foreach ($this->columns() as $oldcol) {

            $id     = $oldcol['id'] ;
            $name   = $oldcol['name'] ;
            $type   = $oldcol['type'] ;
            $source = $oldcol['source'] ;

            $found = false ;
            foreach ($coldef as $col) {
                if (($col->name   == $name) &&
                    ($col->type   == $type) &&
                    ($col->source == $source)) {
                    $found = true ;
                    break ;
                }
            }
            if (!$found) $this->remove_column_by_id ($id) ;
            
        }
        return $this->experiment()->find_run_table_by_id($this->id()) ;
    }
    
    private function add_column_using_ ($col) {
        $is_editable = false ;
        switch ($col->type) {
            case 'Editable':
                $is_editable = true ;
                break ;
            case 'Run Info':
                if ($col->source === 'Run Title') $is_editable = true ;
                break ;
        }
        $this->add_column (
            $col->name ,
            $col->type ,
            $col->source ,
            $is_editable ,
            $col->position) ;
    }
    
    /**
     * Return an export-ready data representation of the table which can
     * be serialized into a JSON object.
     * 
     * @return array
     */
    public function as_data () {
        $coldef = array() ;
        foreach ($this->columns() as $col) {
            array_push($coldef, array (
                'id'          => $col['id'] ,
                'name'        => $col['name'] ,
                'type'        => $col['type'] ,
                'source'      => $col['source'] ,
                'is_editable' => $col['is_editable'] === 'YES' ? 1 : 0 ,
                'position'    => $col['position'])) ;
        }
        $table_data = array (
            'config' => array (
                'id'            => $this->id() ,
                'name'          => $this->name() ,
                'descr'         => $this->descr() ,
                'created_uid'   => $this->created_uid() ,
                'created_time'  => $this->created_time()->toStringShort() ,
                'modified_uid'  => $this->modified_uid() ,
                'modified_time' => $this->modified_time()->toStringShort() ,
                'coldef'        => $coldef
            )
        ) ;
        return $table_data ;
    }
}
?>
