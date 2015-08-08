<?php

namespace RegDB ;

require_once 'regdb.inc.php' ;

/**
 * Class RegDBDataSet an abstraction for the 'psana' dataset specifications.
 *
 * @author gapon
 */
class RegDBDataSet {

    // Object parameters
    private $_dataset ;

    // Data members
    private $_components = null ;   // Parset representation of the data set as
                                    // a dictionary.

    /**
     * Initialize the object. Sanitize undfined or null input by turning
     * it into an empty string.
     *
     * @param String $dataset
     */
    public function __construct ($dataset='') {
        $this->_dataset = isset($dataset) && !is_null($dataset) ? $dataset : '' ;
    }

    /**
     * Return the dataset string
     *
     * @return String
     */
    public function dataset() { return $this->_dataset ; }

    /**
     * Return 'true' if the dataset refers to files at an LCLS FFB filesystem
     *
     * @return Boolean
     */
    public function is_ffb () {
        $dir = $this->get_dir() ;
        return !is_null($dir) && $dir === '/reg/d/ffb/%(instrument_lower)s/%(experiment)s/xtc' ;
    }

    /**
     * Return 'true' if the dataset refers to files at an LCLS ANA filesystem
     *
     * @return Boolean
     */
    public function is_ana () {
        $dir = $this->get_dir() ;
        return is_null($dir) || $dir === '/reg/d/psdm/%(instrument_lower)s/%(experiment)s/xtc' ;
    }

    // 'get' selectors for parameters which must have values
    public function get_exp    () { return $this->_get('exp') ; }
    public function get_run    () { return $this->_get('run') ; }
    public function get_dir    () { return $this->_get('dir') ; }
    public function get_stream () { return $this->_get('stream') ; }

    // 'has' selectors for parameters which don't have values
    public function has_xtc  () { return $this->_has('xtc') ; }
    public function has_smd  () { return $this->_has('smd') ; }
    public function has_idx  () { return $this->_has('idx') ; }
    public function has_h5   () { return $this->_has('h5') ; }
    public function has_live () { return $this->_has('live') ; }

    // public modifiers
    public function set_ffb  () { $this->_set('dir',  '/reg/d/ffb/%(instrument_lower)s/%(experiment)s/xtc') ; }
    public function set_ana  () { $this->_set('dir',  '/reg/d/psdm/%(instrument_lower)s/%(experiment)s/xtc') ; }

    public function set_live    () { $this->_set   ('live') ; }
    public function remove_live () { $this->_remove('live') ; }

    public function set_stream  ($val) {
        $val2parse = isset($val) && !is_null($val) ? trim($val) : '' ;
        foreach (explode(',', $val2parse) as $range) {
            foreach (explode('-', $range) as $str) {
                if (!ctype_digit($str))
                    throw new RegDBException (
                        __METHOD__, 
                        "invalid stream specification '{$val}'") ;
            }
        }
        $this->_set('stream',  $val) ;
    }
    public function remove_stream  () { $this->_remove('stream') ; }

    /**
     * Safe extractor of the parameter value. Return null if the parameter
     * is unknown.
     *
     * @param String $key
     * @return String
     */
    private function _get ($key) {
        $this->_parse() ;
        return array_key_exists($key, $this->_components) ? $this->_components[$key] : null ;
    }
    
    /**
     * Return 'true' if the parsed dataset specification has the key
     * 
     * @param String $key
     * @return Boolean
     */
    private function _has ($key) {
        return !is_null($this->_get($key)) ;
    }

    /**
     * Update dataset component and rebuild teh dataset speification string
     *
     * @param String $key
     * @param String $val
     */
    private function _set ($key, $val=null) {
        $this->_parse() ;
        $this->_components[$key] = isset($val) && !is_null($val) ? $val : '' ;
        $this->_rebuild() ;
    }
    private function _remove($key) {
        $this->_parse() ;
        if (array_key_exists($key, $this->_components)) unset($this->_components[$key]) ;
        $this->_rebuild() ;
    }

    /**
     * Parse dataset string into an internal representation. See specification
     * document at:
     * https://pswww.slac.stanford.edu/swdoc/releases/ana-current/psana-doxy/html/classIData_1_1Dataset.html#_details
     *
     * @throws RegDBException
     */
    private function _parse () {
        if (!is_null($this->_components)) return ;

        foreach (explode(':', $this->_dataset) as $pair) {
            $key = '' ;
            $val = null ;
            $key_val = explode('=', $pair) ;
            switch (count($key_val)) {
                case 2: $val = $key_val[1] ;
                case 1: $key = $key_val[0] ; break ;
                default:
                    throw new RegDBException (
                        __METHOD__, 
                        "invalid dataset specification '{$this->_dataset}'") ;
            }
            switch ($key) {
                case 'exp':
                case 'run':
                case 'dir':
                case 'stream':
                    if (is_null($val))
                        throw new RegDBException (
                            __METHOD__, 
                            "dataset parameter '{$key}' must have a value") ;
                    break ;
                default:
                    // Force others not to have any value even if the one was provided
                    // in the dataset specification.
                    $val = '' ;
            }
            $this->_components[$key] = $val ;
        }
    }

    /**
     * Rebuild dataset specification string from its components.
     * The function is normally called after updating one of the
     * components after an object is already constructed.
     */
    private function _rebuild () {
        $this->_dataset = '' ;
        foreach ($this->_components as $key => $val) {
            if ($this->_dataset !== '') $this->_dataset .= ':' ;
            $this->_dataset .= $key ;
            if ($val !== '') $this->_dataset .= "={$val}" ;
        }
    }
}

/*
// Unit test:

function dump_datasets($datasets) {
    foreach ($datasets as $ds) {
        print <<<HERE
<br>dataset:    <b>{$ds->dataset()}</b>
<br>is_ffb:     <b>{$ds->is_ffb()}</b>
<br>is_ana:     <b>{$ds->is_ana()}</b>
<br>get_dir:    <b>{$ds->get_dir()}</b>
<br>get_stream: <b>{$ds->get_stream()}</b>
<br>has_xtc:    <b>{$ds->has_xtc()}</b>
<br>has_smd:    <b>{$ds->has_smd()}</b>
<br>has_idx:    <b>{$ds->has_idx()}</b>
<br>has_live:   <b>{$ds->has_live()}</b>
<br>
HERE;
    }
}
try {
    $datasets = array (
        new RegDBDataSet('exp=cxi12315:run=2:h5:smd:idx:dir=/reg/d/psdm/%(instrument_lower)s/%(experiment)s/xtc') ,
        new RegDBDataSet('exp=cxi12315:run=2:xtc:smd:live:dir=/reg/d/ffb/%(instrument_lower)s/%(experiment)s/xtc:stream=0-79,81')
    ) ;
    dump_datasets($datasets) ;

    $datasets[0]->set_ffb() ;
    $datasets[0]->set_live() ;
    $datasets[1]->set_ana() ;
    $datasets[1]->remove_live() ;
    $datasets[1]->remove_stream() ;

    $datasets[0]->set_stream('0-1,3,80-81') ;

    dump_datasets($datasets) ;

} catch (Exception $e) {
    print $e ;
}
*/

?>