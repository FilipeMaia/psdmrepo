<?php

namespace websrt ;

require_once 'websrt/websrt.inc.php' ;

class WebSrt {

    public static $types = array ('ana', 'data', 'dm', 'web') ;

    public static $types2prefixes = array (
        'ana'  => 'ana-' ,
        'data' => '' ,
        'dm'   => 'dm-' ,
        'web'  => 'web-'
    ) ;

    public static function  find_release ($relname) {
        $relname = trim($relname) ;
        foreach (scandir(TAGS_DIR) as $name) {
            if ($name === $relname) {
                foreach (WebSrt::$types as $type) {
                    $prefix = WebSrt::$types2prefixes[$type] ;
                    if (preg_match("/^{$prefix}[0-9]+\.[0-9]+\.[0-9]+$/", $name))
                        return new Release($name, $type) ;
                }
                throw new WebSrtException(__METHOD__, "unsupported release type for release: '{$relname}'") ;
            }
        }
        return null ;
    }

    public static function releases($reltype=null) {

        $result = array() ;

        $types2search = array() ;
        if (is_null($reltype)) {
            $types2search = WebSrt::$types ;
        } else {
            if (in_array($reltype, WebSrt::$types)) {
                array_push($types2search, $reltype) ;
            } else {
                throw new WebSrtException(__METHOD__, "unsupported release type: '{$reltype}'") ;
            }
        }

        $files = scandir(TAGS_DIR) ;

        foreach ($types2search as $type) {
            $releases = array() ;
            $prefix = WebSrt::$types2prefixes[$type] ;
            foreach ($files as $name) {
                if (preg_match("/^{$prefix}[0-9]+\.[0-9]+\.[0-9]+$/", $name)) {
                    array_push($releases, new Release($name, $type)) ;
                }
            }
            $result[$type] = WebSrt::sort_and_link($releases) ; ;
        }
        return $result ;
    }

    /**
     * Return a list of srorted and linked releases
     *
     * @param type $releases
     * @return type
     */
    public static function sort_and_link ($releases) {
        $releases = WebSrt::sort_desc($releases) ;
        $num = count($releases) ;
        if ($num > 1)
            for ($i = 0 ; $i < $num - 1 ; $i++)
                $releases[$i]->set_prev_release($releases[$i+1]) ;
        return $releases ;
    }
    
    /**
     * Descending sort of releases by their versions
     *
     * @param array $releases
     * @return array
     */
    public static function sort_desc ($releases) {
        usort($releases, function ($a, $b) {
            $a_ver = $a->version() ;
            $b_ver = $b->version() ;
            return
                $a_ver->as_number() == $b_ver->as_number() ?
                    0 :
                    $a_ver->as_number() < $b_ver->as_number() ?
                        -1 :
                        +1 ;
        }) ;
        return array_reverse($releases) ;
    }
}

?>