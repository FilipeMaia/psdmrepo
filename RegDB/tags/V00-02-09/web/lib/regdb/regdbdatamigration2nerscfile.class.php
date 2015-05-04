<?php

namespace RegDB;

require_once( 'regdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LusiTime\LusiTime;

/**
 * Class RegDBDataMigration2NERSCFile an abstraction for a data file entry found
 * in the NERSC data migration table.
 *
 * @author gapon
 */
class RegDBDataMigration2NERSCFile {

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

    public function name () {
        return $this->attr['file']; }

    public function type () {
        return $this->attr['file_type']; }
        
    public function start_time () {
        return $this->attr['start_time'] ? LusiTime::from64( $this->attr['start_time'] ) : null; }

    public function stop_time () {
        return $this->attr['stop_time'] ? LusiTime::from64( $this->attr['stop_time'] ) : null; }

    public function error_msg () {
        return $this->attr['error_msg']; }
}
?>
