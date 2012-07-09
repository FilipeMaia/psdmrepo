<?php

namespace RegDB;

require_once( 'regdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LusiTime\LusiTime;

/**
 * Class RegDBFile an abstraction for a data file open/created by the DAQ system.
 *
 * @author gapon
 */
class RegDBFile {

    /* Data members
     */
    private $connection;
    private $experiment;

    public $attr;

    /* Constructor
     */
    public function __construct ( $connection, $experiment, $attr ) {
        $this->connection = $connection;
        $this->experiment = $experiment;
        $this->attr = $attr;
    }

    public function parent () {
        return $this->experiment; }

    public function run () {
        return (int)$this->attr['run']; }

    public function stream () {
        return (int)$this->attr['stream']; }

    public function chunk () {
        return (int)$this->attr['chunk']; }
        
    public function open_time () {
        return LusiTime::from64( $this->attr['open'] ); }

    /**
     * Find the migration status of the file if this information is available.
     * Otherwise return null.
     * 
     * NOTES: It's not guaranteed that any information will be found for the file
     * in the data migration database. Normally it's only available for the latest
     * XTC files. And it's not guaranteed to be preserved in the database
     * forever. So, always check for a value returned by the method.
     */
    public function data_migration_file() {
        $table    = "{$this->connection->database}.data_migration";
        $filetype = 'xtc';
        $filename = sprintf("e%d-r%03d-s%02d-c%02d.%s", $this->experiment->id(), $this->run(), $this->stream(), $this-chunk(), $filetype);
        $result = $this->connection->query(
            "SELECT * FROM {$table} WHERE exper_id={$this->experiment()->id()} AND file='{$filename}' AND file_type='{$filetype}'" );

        $nrows = mysql_numrows( $result );
        if( 0 == $nrows ) return null;
        if( 1 != $nrows )
            throw new RegDBException(
                __METHOD__,
                "unexpected size of result set returned by the query; the database may be corrupt or has improper schema" );

        return new RegDBDataMigrationFile (
            $this->connection,
            $this->experiment(),
            mysql_fetch_array( $result, MYSQL_ASSOC ));
   }
   
}
?>
