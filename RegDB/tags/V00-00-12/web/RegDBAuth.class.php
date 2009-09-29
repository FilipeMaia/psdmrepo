<?php
/* 
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * Class RegDBAuth provides an interface to the Authorization Service
 *
 * @author gapon
 */
class RegDBAuth {

    public static function isAuthenticated() {
        return $_SERVER['REMOTE_USER'] != '';
    }
}
?>
