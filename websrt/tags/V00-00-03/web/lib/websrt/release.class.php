<?php

namespace websrt ;

require_once 'websrt/websrt.inc.php' ;

date_default_timezone_set('America/Los_Angeles') ;

class Release {

    // -- parameters --

    private $name = '' ;
    private $type = '' ;
    private $tags_dir = TAGS_DIR ;
    private $deployment_dir = DEPLOYMENT_DIR ;
    private $prev_release = null ;  // -- will be set separatedly, not through the c-tor

    // -- cache --

    private $version = null ;
    private $notes = null ;
    private $status = null ;
    private $tags = null ;

    private $num_modified_pkg = null ;
    private $num_new_pkg = null ;
    private $num_removed_pkg = null ;

    public function __construct ($name, $type) {
        $this->name = $name ;
        $this->type = $type ;
    }

    public function set_prev_release ($release) { $this->prev_release = $release ; }

    public function prev_release () { return $this->prev_release ; }

    public function name () { return $this->name ; }
    public function type () { return $this->type ; }

    public function version () {
        if (is_null($this->version)) {
            $matches = array() ;
            preg_match("/([0-9]+)\.([0-9]+)\.([0-9]+)/", $this->name, $matches) ;
            $this->version = new RelVersion(intval($matches[1]), intval($matches[2]), intval($matches[3])) ;
        }
        return $this->version ;
    }
    
    public function notes () {
        if (is_null($this->notes)) {
            $path = "{$this->tags_dir}/{$this->name}-notes" ;
            $this->notes = file_exists($path) ? file_get_contents($path) : '' ;
        }
        return $this->notes ;
    }
    private function status () {
        if (is_null($this->status)) {
            $this->status = array (
                'deployed_date' => '' ,
                'on_disk'       => false
            ) ;
            $path = "{$this->deployment_dir}/{$this->name}" ;
            if (file_exists($path)) {
                $stat = stat($path) ;
                $this->status['on_disk'] = true ;
            }
            if ($this->status['on_disk']) {
                $path = "{$this->tags_dir}/{$this->name}" ;
                if (file_exists($path)) {
                    $stat = stat($path) ;
                    $this->status['deployed_date'] = strftime("%Y-%m-%d" , $stat['ctime']) ;
                }
            }
        }
        return $this->status ;
    }
    public function deployed_date () {
        $s = $this->status() ;
        return $s['deployed_date'] ;
    }
    public function on_disk () {
        $s = $this->status() ;
        return $s['on_disk'] ;
    }
    public function tags () {
        if (is_null($this->tags)) {
            $this->tags = array() ;
            $path = "{$this->tags_dir}/{$this->name}" ;
            foreach (explode("\n", file_get_contents($path)) as $l) {
                if ($l) {
                    $tag_ver = preg_split ('/[\s\t]+/', $l) ;
                    $this->tags[$tag_ver[0]] = $tag_ver[1] ;
                }
            }
        }
        return $this->tags ;
    }
    private function diff_releases () {

        if (is_null($this->num_modified_pkg) ||
            is_null($this->num_new_pkg) ||
            is_null($this->num_removed_pkg)) {

            $this->num_modified_pkg = 0 ;
            $this->num_new_pkg = 0 ;
            $this->num_removed_pkg = 0 ;

            if ($this->prev_release()) {
                $tags = $this->tags() ;
                $tags_prev = $this->prev_release()->tags() ;
                foreach ($tags as $tag => $ver) {
                    if (array_key_exists($tag, $tags_prev)) {
                        if ($ver != $tags_prev[$tag]) {
                            $this->num_modified_pkg++ ;
                        }
                    } else {
                        $this->num_new_pkg++ ;
                    }
                }
                foreach ($tags_prev as $tag => $ver) {
                    if (!array_key_exists($tag, $tags)) {
                        $this->num_removed_pkg++ ;
                    }
                }
            }
        }        
    }
    public function num_modified_pkg () {
        if (is_null($this->num_modified_pkg)) { $this->diff_releases() ; }
        return $this->num_modified_pkg ;
    }
    public function num_new_pkg () {
        if (is_null($this->num_new_pkg)) { $this->diff_releases() ; }
        return $this->num_new_pkg ;
    }
    public function num_removed_pkg () {
        if (is_null($this->num_removed_pkg)) { $this->diff_releases() ; }
        return $this->num_removed_pkg ;
    }

    /**
     * Return the array representation of the object suitable for converting into JSON.
     *
     * @return array
     */
    public function export2array () {
        return array (
            'name'          => $this->name() ,
            'type'          => $this->type() ,
            'version'       => $this->version()->export2array() ,
            'notes'         => $this->notes() ,
            'tags'          => $this->tags() ,
            'deployed_date' => $this->deployed_date() ,
            'on_disk'       => $this->on_disk() ? 1 : 0
        ) ;
    }
}
?>