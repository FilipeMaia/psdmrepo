<?php

namespace RegDB;

require_once( 'regdb.inc.php' );

require_once( 'authdb/authdb.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use AuthDB\Logger;

use FileMgr\DbConnection;

use LusiTime\LusiTime;

/**
 * Class RegDBConnection encapsulates operations with the database
 */
class RegDBConnection extends DbConnection {

    /* Parameters of the object
     */
    private $ldap_host;
    private $ldap_user;
    private $ldap_passwd;

    /* Current state of the object
     */
    private $ldap_ds;

    /**
     * Constructor
     *
     * The constructor won't make any actual connection attempts. This will be deffered
     * to operations dealing with queries, transactions, etc.
     *
     * @param string $host
     * @param string $user
     * @param string $password
     * @param string $database 
     */
    public function __construct (
        $host,
        $user,
        $password,
        $database,
        $ldap_host,
        $ldap_user,
        $ldap_passwd ) {

        parent::__construct ( $host, $user, $password, $database );

        $this->ldap_host   = $ldap_host;
        $this->ldap_user   = $ldap_user;
        $this->ldap_passwd = $ldap_passwd;
    }

    /**
     * Destructor
     */
    public function __destruct () {
        parent::__destruct();
        if( isset( $this->ldap_ds )) ldap_close( $this->ldap_ds );
    }

    /* ================
     *   LDAP QUERIES
     * ================
     */
    public function posix_groups ( $all_groups=true ) {

        $this->connect();

        // NOTE: It's much safer to generate many simple regular expressions
        //       to be checked individually rather than a single complex
        //       one which may fail due to some internal limitations
        //       of the PHP engine.

        $group_filter = array();
        if( !$all_groups ) {
            $regdb = RegDB::instance();
            $regdb->begin();
            foreach( $regdb->instrument_names() as $instr ) {
                array_push( $group_filter, '/^'.strtolower($instr).'/' );
            }
            array_push( $group_filter, '/^ps/' );
        }
        $list = array();
        $groups = $this->groups();
        foreach( array_keys( $groups ) as $gid ) {
            $name = trim($groups[$gid]['name']);
            if( $group_filter ) {
                foreach( $group_filter as $filter )
                    if ( 1 == preg_match( $filter, $name ))
                        array_push( $list, $name );
            } else {
                array_push( $list, $name );
            }
        }
        sort( $list );
        return $list;
    }

    public function is_known_posix_group ( $name ) {

        $this->connect();

        $trim_name = trim( $name );
        if( $trim_name == '' )
            throw new RegDBException (
                __METHOD__,
               "group name can't be empty" );

        $sr = ldap_search( $this->ldap_ds, "ou=Group,dc=reg,o=slac", "cn=$trim_name" );
        if( !$sr )
            throw new RegDBException (
                __METHOD__,
               "LDAP error: ".ldap_error( $this->ldap_ds ));

        $info = ldap_get_entries( $this->ldap_ds, $sr );

        $num_groups = $info["count"];
        if( $num_groups == 0 ) return false;
        if( $num_groups != 1 )
            throw new RegDBException (
                __METHOD__,
               "inconsistent result set returned from LDAP server" );
        return true;
    }

    public function is_member_of_posix_group ( $group, $uid ) {

        $this->connect();

        $trim_uid = trim( $uid );

        $members = $this->posix_group_members( $group );
        foreach( $members as $member )
            if( $member['uid'] == $trim_uid  ) return true;
        return false;
    }

    public function posix_group_members ( $name, $and_as_primary_group=true ) {

        $this->connect();

        $trim_name = trim( $name );
        if( $trim_name == '' )
            throw new RegDBException (
                __METHOD__,
               "group name can't be empty" );
        $sr = ldap_search( $this->ldap_ds, "ou=Group,dc=reg,o=slac", "cn={$trim_name}" );
        if( !$sr )
            throw new RegDBException (
                __METHOD__,
               "LDAP error: ".ldap_error( $this->ldap_ds ));

        $info = ldap_get_entries( $this->ldap_ds, $sr );

        $num_groups = $info["count"];
        if( $num_groups == 0 )
            throw new RegDBException (
                __METHOD__,
               "no such group known to LDAP server" );
        if( $num_groups != 1 )
            throw new RegDBException (
                __METHOD__,
               "inconsistent result set returned from LDAP server" );

        $result = array();

        // Search members of this secondary group
        //
        if( array_key_exists ( 'memberuid', $info[0] )) {
          for( $i = 0; $i < $info[0]["memberuid"]["count"]; $i++ ) {
              $uid = $info[0]["memberuid"][$i];
              $full_name = "";
              $passwd = posix_getpwnam( $uid );
              if( $passwd ) $full_name = $passwd["gecos"];
              array_push(
                  $result,
                  array(
                      "uid" => $uid,
                      "gecos" => $full_name,
                      "email" => $uid.'@slac.stanford.edu' )
              );
          }
        }

        // (OPTIONALY) Now search user records to see who may claim this group as
        // its primary group.
        //
        if( $and_as_primary_group ) {
            $gid = $info[0]["gidnumber"][0];
            $primary = $this->primary_posix_group_members( $gid );
            foreach( $primary as $p ) {
                array_push(
                    $result,
                    array(
                        "uid"   => $p['uid'],
                        "gecos" => $p['gecos'],
                        "email" => $p['email'] )
                );
            }
        }
        sort( $result );
        return $result;
    }

    public function user_accounts ( $name='*' ) {

        $this->connect();

        $groups = $this->groups();
        $user2groups = $this->user2groups( $groups );

        $trim_name = trim( $name );
        if( $trim_name == '' )
            throw new RegDBException (
                __METHOD__,
               "user name can't be empty" );
        $sr = ldap_search( $this->ldap_ds, "ou=People,dc=reg,o=slac", "uid=$trim_name" );
        if( !$sr )
            throw new RegDBException (
                __METHOD__,
               "LDAP error: ".ldap_error( $this->ldap_ds ));

        $list = array();
        $info = ldap_get_entries( $this->ldap_ds, $sr );
        $num_accounts = $info["count"];
        for( $i = 0; $i < $num_accounts; $i++ ) {
            $uid = $info[$i]["uid"][0];
            $gid_primary = $info[$i]["gidnumber"][0];
            $user_groups = array();
            if( array_key_exists( $uid, $user2groups ))
                $user_groups = $user2groups[$uid];
            array_push( $user_groups, $groups[$gid_primary]['name'] );
            array_push(
                $list,
                array(
                    'uid'   => $uid,
                    'gecos'  => $info[$i]["gecos"][0],
                    'email' => $uid.'@slac.stanford.edu',
                    'groups' => $user_groups
                )
            );
        }
        return $list;
    }

    public function find_user_accounts ( $uid_or_gecos_pattern, $scope ) {

        $this->connect();

        $groups = $this->groups();
        $user2groups = $this->user2groups( $groups );

        $pattern = trim( $uid_or_gecos_pattern );
        if( $pattern == '' )
            throw new RegDBException (
                __METHOD__,
               "search pattern can't be empty" );
        if( $scope["uid"] && $scope["gecos"] ) {
            $filter = "(|(uid=*{$pattern}*)(gecos=*{$pattern}*))";
        } else if( $scope["uid"] ) {
        	$filter = "(uid=*{$pattern}*)";
        } else if( $scope["gecos"] ) {
        	$filter = "(gecos=*{$pattern}*)";
        } else {
        	throw new RegDBException (
                __METHOD__,
               "incorrect search scope" );
        }
        $sr = ldap_search( $this->ldap_ds, "ou=People,dc=reg,o=slac", $filter );
        if( !$sr )
            throw new RegDBException (
                __METHOD__,
               "LDAP error: ".ldap_error( $this->ldap_ds ));

        $list = array();
        $info = ldap_get_entries( $this->ldap_ds, $sr );
        $num_accounts = $info["count"];
        for( $i = 0; $i < $num_accounts; $i++ ) {
            $uid = $info[$i]["uid"][0];
            $gid_primary = $info[$i]["gidnumber"][0];
            $user_groups = array();
            if( array_key_exists( $uid, $user2groups ))
                $user_groups = $user2groups[$uid];
            array_push( $user_groups, $groups[$gid_primary]['name'] );
            array_push(
                $list,
                array(
                    'uid'   => $uid,
                    'gecos'  => $info[$i]["gecos"][0],
                    'email' => $uid.'@slac.stanford.edu',
                    'groups' => $user_groups
                )
            );
        }
        return $list;
    }

    public function find_user_account ( $name ) {

        $this->connect();

        /* Find the account in LDAP. It should exist. Return null if not.
         * Report error if there is nore than one entry.
         */
        $trim_name = trim( $name );
        if( $trim_name == '' )
            throw new RegDBException (
                __METHOD__,
               "user name can't be empty" );
        $sr = ldap_search( $this->ldap_ds, "ou=People,dc=reg,o=slac", "uid={$trim_name}" );
        if( !$sr )
            throw new RegDBException (
                __METHOD__,
               "LDAP error: ".ldap_error( $this->ldap_ds ));

        $info = ldap_get_entries( $this->ldap_ds, $sr );
        $num_accounts = $info["count"];
        if( $num_accounts == 0 ) return null;
        if( $num_accounts != 1 )
            throw new RegDBException (
                __METHOD__,
               "LDAP error: ".ldap_error( $this->ldap_ds ));

        /* Find the primary and secondary groups the user is member of.
         * Either of both can not exist.
         */
        $uid = $info[0]["uid"][0];
        $gid_primary = $info[0]["gidnumber"][0];
        $user_groups = $this->non_primary_groups_for( $uid );
        array_push( $user_groups, $this->gid2name( $gid_primary ));
        return array (
            'uid'    => $uid,
            'gecos'  => $info[0]["gecos"][0],
            'email'  => $uid.'@slac.stanford.edu',
            'groups' => $user_groups );
    }

    /* ATTENTION; This is the fastest implementation for finding non-primary
     * groups the specified user account would belong to.
     */
    public function non_primary_groups_for( $uid ) {
    	$this->connect();
    	$trim_uid = trim( $uid );
        if( $trim_uid == '' )
            throw new RegDBException (
                __METHOD__,
               "UID can't be empty" );
        $sr = ldap_search( $this->ldap_ds, "ou=Group,dc=reg,o=slac", "(&(memberUid={$trim_uid})(objectClass=posixGroup))" );
        if( !$sr )
            throw new RegDBException (
                __METHOD__,
               "LDAP error: ".ldap_error( $this->ldap_ds ));
        $result = array();
        $info = ldap_get_entries( $this->ldap_ds, $sr );
        for( $i = 0; $i < $info["count"]; $i++ ) {
        	array_push( $result, $info[$i]["cn"][0] );
        }
        return $result;
    }
    
    public function gid2name( $gid ) {
    	$this->connect();
    	$trim_gid = trim( $gid );
        if( $trim_gid == '' )
            throw new RegDBException (
                __METHOD__,
               "GID can't be empty" );
        $sr = ldap_search( $this->ldap_ds, "ou=Group,dc=reg,o=slac", "(&(gidNumber={$trim_gid})(objectClass=posixGroup))" );
        if( !$sr )
            throw new RegDBException (
                __METHOD__,
               "LDAP error: ".ldap_error( $this->ldap_ds ));
        $info = ldap_get_entries( $this->ldap_ds, $sr );
        $num_accounts = $info["count"];
        if( $num_accounts != 1 )
            throw new RegDBException (
                __METHOD__,
               "Inconsistent result reported by LDAP server" );
        return $info[0]["cn"][0];
    }

    public function primary_posix_group_members ( $gid ) {

        $this->connect();

        $groups = $this->groups();
        $user2groups = $this->user2groups( $groups );

        $sr = ldap_search( $this->ldap_ds, "ou=People,dc=reg,o=slac", "(&(uid=*)(gidnumber=".$gid."))" );
        if( !$sr )
            throw new RegDBException (
                __METHOD__,
               "LDAP error: ".ldap_error( $this->ldap_ds ));

        $list = array();
        $info = ldap_get_entries( $this->ldap_ds, $sr );
        $num_accounts = $info["count"];
        for( $i = 0; $i < $num_accounts; $i++ ) {
            $uid = $info[$i]["uid"][0];
            $gid_primary = $info[$i]["gidnumber"][0];
            $user_groups = array();
            if( array_key_exists( $uid, $user2groups ))
                $user_groups = $user2groups[$uid];
            array_push( $user_groups, $groups[$gid_primary]['name'] );
            array_push(
                $list,
                array(
                    'uid'   => $uid,
                    'gecos'  => $info[$i]["gecos"][0],
                    'email' => $uid.'@slac.stanford.edu',
                    'groups' => $user_groups
                )
            );
        }
        return $list;
    }

    private function user2groups( $groups ) {
        $list = array();
        foreach( array_keys($groups) as $gid ) {
            $name = $groups[$gid]['name'];
            foreach( $groups[$gid]['members'] as $uid ) {
                $user_groups = array_key_exists( $uid, $list ) ? $list[$uid] : array();
                array_push( $user_groups, $name );
                $list[$uid] = $user_groups;
            }
        }
        return $list;
    }
    private function groups ( $gid2name=true ) {
        $this->connect();

        $sr = ldap_search( $this->ldap_ds, "ou=Group,dc=reg,o=slac", "cn=*" );
        if( !$sr )
            throw new RegDBException (
                __METHOD__,
               "LDAP error: ".ldap_error( $this->ldap_ds ));

        $result = array();

        $info = ldap_get_entries( $this->ldap_ds, $sr );
        for( $i = 0; $i < $info["count"]; $i++ ) {
            $members = array();
            if( array_key_exists ( 'memberuid', $info[$i] )) {
              for( $j = 0; $j < $info[$i]["memberuid"]["count"]; $j++ )
                  array_push( $members, $info[$i]["memberuid"][$j] );
            }

            if( $gid2name )
                $result[$info[$i]["gidnumber"][0]] = array(
                    'name' => $info[$i]["cn"][0],
                    'members' => $members
                );
            else
                $result[$info[$i]["cn"][0]] = array(
                    'gid' => $info[$i]["gidnumber"][0],
                    'members' => $members
                );
        }
        return $result;
    }

    /**
     * Make a connection if this hasn't been done yet.
     */
    protected function connect () {
//        if( !isset( $this->link )) {
//
//            /* Connect to MySQL server and register the shutdown function
//             * to clean up after self by doing 'ROLLBACK'. This would allow
//             * to use so called 'persistent' MySQL connections.
//             * 
//             * NOTE: using the 'persistent' connection. This connection won't be
//             * closed by 'mysql_close()'.
//             */
//            $new_link = false; // true;
//            $this->link = mysql_pconnect( $this->host, $this->user, $this->password, $new_link );
//            if( !$this->link )
//                throw new RegDBException (
//                    __METHOD__,
//                    "MySQL error: ".mysql_error( $this->link ).", in function: mysql_connect" );
//
//            if( !mysql_select_db( $this->database, $this->link ))
//                throw new RegDBException (
//                    __METHOD__,
//                    "MySQL error: ".mysql_error( $this->link ).", in function: mysql_select_db" );
//
//            $sql = "SET SESSION SQL_MODE='ANSI'";
//            if( !mysql_query( $sql, $this->link ))
//                throw new RegDBException (
//                    __METHOD__,
//                    "MySQL error: ".mysql_error( $this->link ).', in query: '.$sql );
//
//            $sql = "SET SESSION AUTOCOMMIT=0";
//            if( !mysql_query( $sql, $this->link ))
//                throw new RegDBException (
//                    __METHOD__,
//                    "MySQL error: ".mysql_error( $this->link ).', in query: '.$sql );
//
//            register_shutdown_function( array( $this, "rollback" ));
//        }

        parent::connect();

        if( !isset( $this->ldap_ds )) {
        
            /* Connect to LDAP server
             */
            $this->ldap_ds = ldap_connect( /* $this->ldap_host */ );
            if( !$this->ldap_ds )
                throw new RegDBException (
                    __METHOD__,
                    "failed to connect to LDAP server: ".$this->ldap_host );

            /* IMPORTANT: This operation is required. Otherwise LDAP client may
             * get confused and attempt guessing a wrong version of protocol before
             * producing a usefull result. A problem is that the first failed attempt to use
             * protocol version 2 will result in a warning message which would destorte
             * the JSON output.
             */
            ldap_set_option( $this->ldap_ds, LDAP_OPT_PROTOCOL_VERSION, 3 );

            if( !ldap_bind( $this->ldap_ds ))
                throw new RegDBException (
                    __METHOD__,
                   "failed to bind to LDAP server due to: ".ldap_error( $this->ldap_ds ));
        }
    }


   /* ================================
     *   LDAP MODIFICATION REQUESTS
     * ===============================
     *
     * ATTENTION: These requests are made on a separate LDAP connection
     * using SASL bind method which requires a special account and MD5 encripted
     * password. The connection is open just temporarily for a duration of each 
     * function call. So, be aware about possible performance implication of
     * these operations!
     */

    public function add_user_to_posix_group ( $user_name, $group_name ) {
        $this->posix_group_op( 'add', $user_name, $group_name );
    }

    public function remove_user_from_posix_group ( $user_name, $group_name ) {
        $this->posix_group_op( 'delete', $user_name, $group_name );
    }
    private function protected_connect2ldap () {

        /* Connect to LDAP server
         */
        $ldap_ds = ldap_connect( /*$this->ldap_host*/ );
        if( !$ldap_ds )
            throw new RegDBException (
                __METHOD__,
                "failed to connect to LDAP server: ".$this->ldap_host );

        /* IMPORTANT: This operation is required. Otherwise LDAP client may
         * get confused and attempt guessing a wrong version of protocol before
         * producing a usefull result. A problem is that the first failed attempt to use
         * protocol version 2 will result in a warning message which would destorte
         * the JSON output.
         */
        ldap_set_option( $ldap_ds, LDAP_OPT_PROTOCOL_VERSION, 3 );

        if( !ldap_sasl_bind ( $ldap_ds, NULL, $this->ldap_passwd, 'digest-md5', NULL, $this->ldap_user ))
        throw new RegDBException (
                __METHOD__,
               "failed to bind to LDAP server due to: ".ldap_error( $ldap_ds ));

        return $ldap_ds;
    }
    private function posix_group_op ( $opname, $user_name, $group_name ) {

        if(( $opname != 'add' ) && ( $opname != 'delete' ))
            throw new RegDBException (
                __METHOD__,
               "internal error: unknown operation name: ".$opname );

        $trim_user_name  = trim( $user_name );
        if( $trim_user_name == '' )
            throw new RegDBException (
                __METHOD__,
               "user name can't be empty" );

        $trim_group_name = trim( $group_name );
        if( $trim_group_name == '' )
            throw new RegDBException (
                __METHOD__,
               "group name can't be empty" );

        $memberuid['memberUid'] = $trim_user_name;
        $op = 'ldap_mod_'.substr ( $opname, 0, 3 );

        $ldap_ds = $this->protected_connect2ldap();
        if(      $opname == 'add'    ) $sr = $op( $ldap_ds, "cn={$trim_group_name},ou=Group,dc=reg,o=slac", $memberuid );
        else if( $opname == 'delete' ) $sr = $op( $ldap_ds, "cn={$trim_group_name},ou=Group,dc=reg,o=slac", $memberuid );
        else {
            ldap_close( $ldap_ds );
            throw new RegDBException (
                __METHOD__,
               "Internal error - illegal LDAP operation requested: ".$op );
        }
        if( $sr == False ) {
            $error = ldap_error( $ldap_ds );
       	    ldap_close( $ldap_ds );
            throw new RegDBException (
                __METHOD__,
               "LDAP error: ".$error );
        }
        ldap_close( $ldap_ds );
        
        /* Report the operation to the logging facility
         */
        $logger = Logger::instance();
        $logger->begin();
        $logger->group_management( $opname, $user_name, $group_name );
        $logger->commit();
    }
}

/* ==========================
 * UNIT TEST FOR LDAP METHODS
 * ==========================
 *

require_once( "regdb.inc.php");

$conn = new RegDBConnection (
    REGDB_DEFAULT_HOST,
    REGDB_DEFAULT_USER,
    REGDB_DEFAULT_PASSWORD,
    REGDB_DEFAULT_DATABASE,
    REGDB_DEFAULT_LDAP_HOST,
    REGDB_DEFAULT_LDAP_USER,
    REGDB_DEFAULT_LDAP_PASSWD );

try {
	print "<br>";
	print "<b>Testing gid2name()</b>";
    print "<br>";
	foreach( array(1013, 1109, 2306) as $gid ) {
		$group = $conn->gid2name( $gid );
	    print "<br>".$gid." : ".$group;
	}
	print "<br>";
	print "<br><b>Testing find_user_account() to get a list of groups the user belongs to</b>";

	foreach( array('gapon', 'sxropr', 'rhabura', 'amo14410') as $user) {
        print "<br>";
		$account = $conn->find_user_account($user);
        print("<br>user=<b>".$account['uid']."</b>");
        foreach( $account['groups'] as $g ) {
            print("<br>".$g);
        }
    }


    $conn->remove_user_from_posix_group( "gapon", "sxr11410" );
    $conn->remove_user_from_posix_group( "salnikov", "sxr11410" );

    $user = $conn->find_user_account( "gapon" );
    print_r( $user );

	$group = $conn->gid2name( 1013 );
	print "1013 : ".$group;

    $list = $conn->user_accounts( "gapon" );
    print_r( $list );
    $list = $conn->posix_group_members( "ec" );
    print_r( $list );
    $groups = $conn->groups();
    print( "<br>Total of ".count($groups)." groups found" );
    $gids = array_keys( $groups );
    sort( $gids, SORT_NUMERIC );
    foreach( $gids as $gid ) {
        print( '<br>'.$gid." : ".$groups[$gid]['name'] );
        print( ', members: ' );
        foreach( $groups[$gid]['members'] as $m )
            print( $m." " );
    }
    echo "<br><hr>";
    $accounts = $conn->user_accounts();
    sort( $accounts );
    print( "<br>Total of ".count($accounts)." user account found" );
    foreach( $accounts as $account ) {
        echo "<br><hr>";
        print_r( $account );
    }

    $groups = $conn->posix_groups();
    foreach( $groups as $group ) {
        $members = $conn->posix_group_members( $group );
        echo "<br><hr>Group: <b>" . $group."</b>";
        echo "<br>&nbsp;&nbsp;Members: ";
        foreach( $members as $member )
            echo "<br>&nbsp;&nbsp;&nbsp;&nbsp;<b>".$member["uid"]."</b>&nbsp;".$member["gecos"];
    }

    echo "<br><hr>";
    $test_groups = array_merge( $groups, array( "aaa" ));
    foreach( $test_groups as $group ) {
        if( $conn->is_known_posix_group( $group ))
            echo "<br>Group <b>$group</b> is known";
        else
            echo "<br>Group <b>$group</b> is NOT known";
    }

 } catch ( RegDBException $e ) {
    print( $e->toHtml());
}
*
*/
?>
