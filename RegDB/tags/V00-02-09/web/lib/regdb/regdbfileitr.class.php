<?php

namespace RegDB ;

require_once 'regdb.inc.php' ;


/**
 * Class RegDBFileItr an iterator of files
 *
 * Initializing the current value is required because the iterator
 * will be used:
 *
 *   while valid() do
 *      key()
 *      current()
 *      next()
 *   end
 *
 * @author gapon
 */
class RegDBFileItr implements \Iterator {

    // Object parameters

    private $regdb = null ;
    private $connection = null ;
    private $sql = null ;

    // Iterator context

    private $result = null ;
    private $nrows = null ;
    private $position = null ;
    private $current_value = null ;

    // Optimization cache

    private $experiments = array() ;
    
    /**
     * Constructor
     *
     * @param \RegDB\RegDB $regdb
     * @param \RegDB\RegDBConnection $connection
     * @param string $sql
     */
    public function __construct ($regdb, $connection, $sql) {
        $this->regdb = $regdb ;
        $this->connection = $connection ;
        $this->sql = $sql ;
    }

    /**
     * Reset the iterator context back to its initial state
     */
    public function rewind () {
        $this->result = null ;
        $this->nrows = null ;
        $this->position = null ;
        $this->current_value = null ;
    }

    /**
     * Return an experiment object
     *
     * @param integer $exper_id
     * @return \RegDB\RegDBExperiment
     * @throws RegDBException
     */
    private function _experiment ($exper_id) {
        if (!array_key_exists($exper_id, $this->experiments)) {
            $this->experiments[$exper_id] = $this->regdb->find_experiment_by_id($exper_id) ;
            if (is_null($this->experiments[$exper_id]))
                throw new RegDBException (
                    __METHOD__ ,
                    "no such experiment: '{$exper_id}'") ;
        }
        return $this->experiments[$exper_id] ;
    }

    /**
     * Return the current element the iterator is set to
     *
     * @return \RegDB\RegDBFile
     */
    public function current () {

        $this->_init() ;

        // Lazy initialization

        if (is_null($this->current_value)) {
            $attr = mysql_fetch_array($this->result, MYSQL_ASSOC) ;
            $exper_id = intval($attr['exper_id']) ;
            $this->current_value = new RegDBFile (
                $this->connection ,
                $this->_experiment($exper_id) ,
                $attr) ;
        }
        return $this->current_value ;
    }

    /**
     * Return a unique key of the current element
     *
     * @return integer
     */
    public function key () {
        $this->_init() ;
        return $this->position ;
    }

    /**
     * Advance the iterator to the next element of the collection
     */
    public function next() {
        $this->_init() ;
        ++$this->position ;
        $this->current_value = null ;   // will be automatically initialized by
                                        // method current()
    }
    
    /**
     * Return the status of the iterator. Return 'false' if the collection
     * is empty or the current pointer is past end of the collection.
     *
     * @return boolean
     */
    public function valid () {
        $this->_init() ;
        return $this->position < $this->nrows ;
    }

    /**
     * Initialize iterator context if needed
     */
    private function _init () {
        if (is_null($this->result)) {
            $this->result = $this->connection->query($this->sql) ;
            $this->nrows = mysql_numrows($this->result) ;
            $this->position = 0 ;
        }
    }
}

?>