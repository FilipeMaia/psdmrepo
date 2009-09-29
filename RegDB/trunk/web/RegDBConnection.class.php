<?php

/**
 * Class RegDBConnection encapsulates operations with the database
 */
class RegDBConnection {

    /* Parameters of the object
     */
    private $host;
    private $user;
    private $password;
    private $database;
    private $ldap_host;

    /* Current state of the object
     */
    private $link;
    private $ldap_ds;
    private $in_transaction = false;

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
    public function __construct ( $host, $user, $password, $database, $ldap_host ) {
        $this->host      = $host;
        $this->user      = $user;
        $this->password  = $password;
        $this->database  = $database;
        $this->ldap_host = $ldap_host;
    }

    /**
     * Destructor
     */
    public function __destruct () {
        if( isset( $this->link    )) mysql_close( $this->link );
        if( isset( $this->ldap_ds )) ldap_close( $this->ldap_ds );
    }

    /*
     * ================================
     *   MySQL TRANSACTION MANAGEMENT
     * ================================
     */
    public function begin () {
        $this->connect();
        if( $this->in_transaction ) return;
        $this->transaction( 'BEGIN' );
        $this->in_transaction = true;
    }

    public function commit () {
        $this->connect();
        if( !$this->in_transaction ) return;
        $this->transaction( 'COMMIT' );
        $this->in_transaction = false;
    }

    public function rollback () {
        $this->connect();
        if( !$this->in_transaction ) return;
        $this->transaction( 'ROLLBACK' );
        $this->in_transaction = false;
    }

    private function transaction ( $transition ) {
        if( !mysql_query( $transition, $this->link ))
            throw new RegDBException (
                __METHOD__,
                "MySQL error: ".mysql_error( $this->link ).', in query: '.$transition );
    }

    /* =================
     *   MySQL QUERIES
     * =================
     */
    public function query ( $sql ) {

        $this->connect();

        if( !$this->in_transaction )
            throw new RegDBException (
                __METHOD__, "no active transaction" );

        $result = mysql_query( $sql, $this->link );
        if( !$result )
            throw new RegDBException (
                __METHOD__,
                "MySQL error: ".mysql_error( $this->link ).', in query: '.$sql );
        return $result;
    }

    /* ==========================================================
     *   MISC. OPERATIONS REQUIREING DATABASE SERVER CONNECTION
     * ==========================================================
     */
    public function escape_string( $text ) {
        return mysql_real_escape_string( $text, $this->link );  }

    /* ================
     *   LDAP QUERIES
     * ================
     */
    public function posix_groups ( ) {

        $this->connect();

        $list = array();
        $groups = $this->groups();
        foreach( array_keys( $groups ) as $gid )
            array_push( $list, $groups[$gid]['name'] );

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

    public function posix_group_members ( $name ) {

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

        // Now search user records to see who may claim this group as
        // its primary group.
        //
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
               "group name can't be empty" );
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

        $list = array();
        $info = ldap_get_entries( $this->ldap_ds, $sr );
        $num_groups = $info["count"];
        for( $i = 0; $i < $num_groups; $i++ ) {
            $members = array();
            $num_members = $info[$i]["memberuid"]["count"];
            for( $j = 0; $j < $num_members; $j++ )
                array_push( $members, $info[$i]["memberuid"][$j] );

            if( $gid2name )
                $list[$info[$i]["gidnumber"][0]] = array(
                    'name' => $info[$i]["cn"][0],
                    'members' => $members
                );
            else
                $list[$info[$i]["cn"][0]] = array(
                    'gid' => $info[$i]["gidnumber"][0],
                    'members' => $members
                );
        }
        return $list;
    }

    /**
     * Make a connection if this hasn't been done yet.
     */
    private function connect () {
        if( !isset( $this->link )) {

            /* Connect to MySQL server
             */
            $new_link = true;
            $this->link = mysql_connect( $this->host, $this->user, $this->password, $new_link );
            if( !$this->link )
                throw new RegDBException (
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).", in function: mysql_connect" );

            if( !mysql_select_db( $this->database, $this->link ))
                throw new RegDBException (
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).", in function: mysql_select_db" );

            $sql = "SET SESSION SQL_MODE='ANSI'";
            if( !mysql_query( $sql, $this->link ))
                throw new RegDBException (
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).', in query: '.$sql );

            $sql = "SET SESSION AUTOCOMMIT=0";
            if( !mysql_query( $sql, $this->link ))
                throw new RegDBException (
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).', in query: '.$sql );

            /* Connect to LDAP server
             */
            $this->ldap_ds = ldap_connect( $this->ldap_host );
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
}

/* ==========================
 * UNIT TEST FOR LDAP METHODS
 * ==========================
 *

require_once( "RegDB.inc.php");

$conn = new RegDBConnection (
    REGDB_DEFAULT_HOST,
    REGDB_DEFAULT_USER,
    REGDB_DEFAULT_PASSWORD,
    REGDB_DEFAULT_DATABASE,
    REGDB_DEFAULT_LDAP_HOST );

try {
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
    print( e.toHtml());
}
*/
?>
