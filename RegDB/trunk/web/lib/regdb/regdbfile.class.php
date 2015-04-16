<?php

namespace RegDB ;

require_once 'regdb.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;

/**
 * Class RegDBFile an abstraction for a data file open/created by the DAQ system.
 *
 * @author gapon
 */
class RegDBFile {

    // Object parameters

    private $connection ;
    private $experiment ;

    public $attr ;

    /* Constructor
     */
    public function __construct ($connection, $experiment, $attr) {
        $this->connection = $connection ;
        $this->experiment = $experiment ;
        $this->attr = $attr ;
    }

    public function parent     () { return $this->experiment ; }
    public function experiment () { return $this->experiment ; }
    public function exper_id   () { return intval($this->attr['exper_id']) ; }
    public function run        () { return intval($this->attr['run']) ; }
    public function stream     () { return intval($this->attr['stream']) ; }
    public function chunk      () { return intval($this->attr['chunk']) ; }
    public function open_time  () { return LusiTime::from64($this->attr['open']) ; }
    public function base_name  () { return  $filename = sprintf("e%d-r%04d-s%02d-c%02d", $this->exper_id(), $this->run(), $this->stream(), $this->chunk()) ; }

    /**
     * Find the migration status of the file at the specified stage
     * if this information is available. Otherwise return null.
     * 
     * NOTE: It's not guaranteed that any information will be found for the file
     *       in the data migration database. Normally it's only available for the latest
     *       XTC files. And it's not guaranteed to be preserved in the database
     *       forever. So, always check for a value returned by the method.
     */
    public function data_migration_file ($stage='') {
        
        if (!in_array($stage, array('','ana','nersc')))
            throw new RegDBException (
                __METHOD__,
                "the specified migration stage '{$stage}' is not supported") ;

        $table = "{$this->connection->database}.data_migration".($stage == '' ? '' : '_'.$stage) ;
        $filetype = 'xtc' ;
        $filename = $this->connection->escape_string(sprintf("e%d-r%04d-s%02d-c%02d.%s", $this->exper_id(), $this->run(), $this->stream(), $this->chunk(), $filetype)) ;
        $sql = "SELECT * FROM {$table} WHERE exper_id={$this->exper_id()} AND file='{$filename}'" ;
        $result = $this->connection->query( $sql ) ;

        $nrows = mysql_numrows($result) ;
        if (0 == $nrows) return null ;
        if (1 != $nrows)
            throw new RegDBException (
                __METHOD__ ,
                "unexpected size of result set returned by the query; the database may be corrupt or has improper schema") ;

        return $stage == 'nersc' ?
            new RegDBDataMigration2NERSCFile($this->connection, $this->experiment(), mysql_fetch_array($result, MYSQL_ASSOC)) :
            new RegDBDataMigrationFile      ($this->connection, $this->experiment(), mysql_fetch_array($result, MYSQL_ASSOC)) ;
   }
}
?>
