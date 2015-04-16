<?php

namespace DataPortal;

require_once( 'dataportal.inc.php' );
require_once( 'authdb/authdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use AuthDB\AuthDB;
use LusiTime\LusiTime;

/* The utility class to generate HTML documents adhering to the standard
 * look & feel. Here is how it should be used to generate a document:
 *
 *   <?php
 *     require_once('dataportal/dataportal.inc.php');
 *     DataPortal::begin(<page name>);
 *   ?>
 *
 *   [ page-specific CSS ]
 *
 *   <?php DataPortal::scripts(); ?>
 *
 *   [ page-specific JavaScript ]
 *
 *   <?php DataPortal::body(); ?>
 *
 *   [ page specific html to go into the document body ]
 *
 *   <?php DataPortal::end(); ?>
 */
class DataPortal {

    /* --------------------------------------------------------------------------------------------
     */
    static function begin( $page_name ) {
        echo <<<HERE

<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html> 
<head> 
<title>{$page_name}</title> 
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"> 

<link type="text/css" href="/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css" rel="Stylesheet" />

<link type="text/css" href="../portal/css/default.css" rel="Stylesheet" />

HERE;
    }

    /* --------------------------------------------------------------------------------------------
     */
    static function scripts( $page_specific_init  ) {

        $auth_svc = AuthDB::instance();
        $auth_svc->begin();
        
        echo <<<HERE

<script type="text/javascript" src="/jquery/js/jquery.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui-1.9.1.custom.min.js"></script>
        
<script type="text/javascript" src="../webfwk/js/Utilities.js"></script>
<script type="text/javascript">

/* ----------------------------------------------
 * Window geometry (compatible with all browsers)
 * ----------------------------------------------
 */
function document_inner_geometry() {
    var winW = 630, winH = 460;
    if (document.body && document.body.offsetWidth) {
    winW = document.body.offsetWidth;
    winH = document.body.offsetHeight;
    }
    if (document.compatMode=='CSS1Compat' &&
        document.documentElement &&
        document.documentElement.offsetWidth ) {
        winW = document.documentElement.offsetWidth;
        winH = document.documentElement.offsetHeight;
    }
    if (window.innerWidth && window.innerHeight) {
        winW = window.innerWidth;
        winH = window.innerHeight;
    }
    var result = {width: winW, height: winH };
    return result;
}
/* ----------------------------------------
 * Authentication and authorization context
 * ----------------------------------------
 */
var auth_is_authenticated="{$auth_svc->isAuthenticated()}";
var auth_type="{$auth_svc->authType()}";
var auth_remote_user="{$auth_svc->authName()}";

var auth_webauth_token_creation="{$_SERVER['WEBAUTH_TOKEN_CREATION']}";
var auth_webauth_token_expiration="{$_SERVER['WEBAUTH_TOKEN_EXPIRATION']}";

function refresh_page() {
    window.location = "{$_SERVER['REQUEST_URI']}";
}

/*
 * Session expiration timer for WebAuth authentication.
 */
var auth_timer = null;
function auth_timer_restart() {
    if( auth_is_authenticated && ( auth_type == 'WebAuth' ))
        auth_timer = window.setTimeout( 'auth_timer_event()', 1000 );
}
var auth_last_secs = null;
function auth_timer_event() {

    var auth_expiration_info = document.getElementById( "auth_expiration_info" );
    var now = mktime();
    var seconds = auth_webauth_token_expiration - now;
    if( seconds <= 0 ) {
        $('#popupdialogs').html(
            '<p><span class="ui-icon ui-icon-alert" style="float:left;"></span>'+
            'Your WebAuth session has expired. Press <b>Ok</b> or use <b>Refresh</b> button'+
            'of the browser to renew your credentials.</p>'
        );
        $('#popupdialogs').dialog({
            resizable: false,
            modal: true,
            buttons: {
                "Ok": function() {
                    $( this ).dialog( "close" );
                    refresh_page();
                }
            },
            title: 'Session Expiration Notification'
        });
        return;
    }
    var hours_left   = Math.floor(seconds / 3600);
    var minutes_left = Math.floor((seconds % 3600) / 60);
    var seconds_left = Math.floor((seconds % 3600) % 60);

    var hours_left_str = hours_left;
    if( hours_left < 10 ) hours_left_str = '0'+hours_left_str;
    var minutes_left_str = minutes_left;
    if( minutes_left < 10 ) minutes_left_str = '0'+minutes_left_str;
    var seconds_left_str = seconds_left;
    if( seconds_left < 10 ) seconds_left_str = '0'+seconds_left_str;

    auth_expiration_info.innerHTML=
        '<b>'+hours_left_str+':'+minutes_left_str+'.'+seconds_left_str+'</b>';

    auth_timer_restart();
}

function logout() {
    $('#popupdialogs').html(
        '<p><span class="ui-icon ui-icon-alert" style="float:left;"></span>'+
        'This will log yout out from the current WebAuth session. Are you sure?</p>'
     );
    $('#popupdialogs').dialog( {
        resizable: false,
        modal: true,
        buttons: {
            "Yes": function() {
                $( this ).dialog( "close" );
                document.cookie = 'webauth_wpt_krb5=; expires=Fri, 27 Jul 2001 02:47:11 UTC; path=/';
                document.cookie = 'webauth_at=; expires=Fri, 27 Jul 2001 02:47:11 UTC; path=/';
                refresh_page();
            },
            Cancel: function() {
                $( this ).dialog( "close" );
            }
        },
        title: 'Session Logout Warning'
    } );
}

function edit_dialog( title, msg, on_save, on_cancel, width, height ) {
    $('#editdialogs').html(msg);
    $('#editdialogs').dialog({
        title:     title,
        resizable: true,
        width:     width == null ? 460 : width,
        height:    height == null ? 240 : height,
        modal:     true,
        buttons:   {
            Save: function() {
                $(this).dialog('close');
                if( on_save != null ) on_save();
            },
            Cancel: function() {
                $(this).dialog('close');
                if( on_cancel != null ) on_cancel();
            }
        }
    });
}

function large_dialog( title, msg ) {
    var geom = document_inner_geometry();
    $('#largedialogs').html(msg);
    $('#largedialogs').dialog({
        title:     title,
        resizable: true,
        width:     geom.width-20,
        height:    geom.height-20,
        modal:     true
    });
}

function report_error( title, msg ) {
    $('#popupdialogs').html(
        '<p><span class="ui-icon ui-icon-alert" style="float:left;"></span>'+
        msg+
        '</p>'
    );
    $('#editdialogs').dialog({
        title:     title,
        resizable: true,
        modal:     true
    });
}
    
/* --------------------------------------------------- 
 * The starting point where the JavaScript code starts
 * ---------------------------------------------------
 */
$(document).ready(
    function(){
        auth_timer_restart();
        {$page_specific_init}();
    }
);
</script>

HERE;
    }

    /* --------------------------------------------------------------------------------------------
     */
    static function body( $document_title, $document_subtitle=null, $document_subtitle_id=null, $extra=null ) {

        require_once('authdb/authdb.inc.php');

        $auth_svc = AuthDB::instance();
        $auth_svc->begin();

        echo <<<HERE
</head>
<body>
  <div id="body">
  <div id="header">
    <div style="float:left;">
      <span class="document_title">{$document_title}</span>
      <span class="document_subtitle" id="{$document_subtitle_id}">{$document_subtitle}</span>
    </div>
    <div style="float:left;">{$extra}</div>
    <div style="float:right;">
      <table><tbody><tr>
        <td valign="bottom">
          <div style="float:right; margin-right:10px;" class="not4print"><a href="javascript:printer_friendly('tabs-experiment')" title="Printer friendly version of this page"><img src="../portal/img/PRINTER_icon.gif" /></a></div>
          <!--
          <div style="float:right; margin-right:10px;" class="not4print"><a href="javascript:pdf('experiment')" title="PDF version of this page"><img src="../portal/img/PDF_icon.gif" /></a></div>
          -->
          <div style="clear:both;" class="not4print"></div>
        </td>
        <td>
          <table class="login"><tbody>
            <tr>
              <td>&nbsp;</td>
              <td>[<a href="javascript:logout()" title="close the current WebAuth session">logout</a>]</td>
            </tr>
            <tr>
              <td>Welcome,&nbsp;</td>
              <td><p><b>{$auth_svc->authName()}</b></p></td>
            </tr>
            <tr>
              <td>Session expires in:&nbsp;</td>
              <td><p id="auth_expiration_info"><b>00:00.00</b></p></td>
            </tr>
          </tbody></table>
        </td>
      </tr></tbody></table>
    </div>
    <div style="clear:both;"></div>
  </div>

HERE;
    }

    /* --------------------------------------------------------------------------------------------
     */
    static function tabs( $tabs_id, $tabs  ) {
        echo DataPortal::tabs_html( $tabs_id, $tabs );
    }
    static function tabs_html( $tabs_id, $tabs  ) {
        $html = <<<HERE
<div id="{$tabs_id}">
  <ul>

HERE;
        foreach( $tabs as $t  ) {
            $id = $t['id'];
                        $style_html = isset($t['separate']) ? 'style="margin-left:10px;" ' : '';
            $href = isset($id) ? '#'.$id : $t['url'];
            $name = $t['name'];
            $onclick = '';
            if( array_key_exists ( 'callback', $t )) $onclick = "onclick='".$t['callback']."'";
            $html .= <<<HERE
    <li {$style_html} ><a href="{$href}" {$onclick}>{$name}</a></li>

HERE;
        }
        $html .= <<<HERE
  </ul>

HERE;
        foreach( $tabs as $t  ) {
            $id = $t['id'];
            if(!isset($id)) continue;
            $tab_html = $t['html'];
            $class    = $t['class'];
            $class_html = ( isset( $class ) && !is_null( $class ) && ( $class != '' )) ? 'class="'.$class.'"' : '';

            // NOTE: the inner <div> is needed to have full control over custom styles,
            //       classes, etc. That's because JQuery UI will manage its own classes
            //       and styles to the id-entified outer <div>.
            //
            $html .= <<<HERE
<div id="{$id}">
  <div {$class_html}>{$tab_html}</div>
</div>

HERE;
        }
        $html .= <<<HERE
</div>

HERE;
        return $html;
    }
    
    /* ---------------------------------------------------------------------------------------------
     */
    static function end() {

        echo <<<HERE

  <div id="popupdialogs" style="display:none;"></div>
  <div id="editdialogs"  style="display:none;"></div>
  <div id="largedialogs"  style="display:none; overflow:auto;"></div>
  <div id="forms" style="display:none;"></div>
  
</div>
</body>
</html>

HERE;

    }

    /* ---------------------------------------------------------------------------------------------
     */
    static function table_begin_html( $cols ) {

        $html = <<<HERE
<table cellspacing="4">
  <tbody>
    <tr>

HERE;

        foreach( $cols as $c ) {
            $style  = array_key_exists( 'style',  $c ) ? 'style="'. $c['style']. '"' : '';
            $width  = array_key_exists( 'width',  $c ) ? 'width="'. $c['width']. '"' : '';
            $align  = array_key_exists( 'align',  $c ) ? 'align="'. $c['align']. '"' : '';
            $valign = array_key_exists( 'valign', $c ) ? 'valign="'.$c['valign'].'"' : '';
            $name   = array_key_exists( 'name',   $c ) ? $c['name']                  : '&nbsp;';
            $html .= <<<HERE
      <td class="table_hdr" {$style} {$width} {$align} {$valign}>{$name}</td>

HERE;
        }
        $html .= <<<HERE
    </tr>

HERE;

        return $html;
    }

    static function table_row_html( $values, $end_of_group=true, $td_options=array(), $tr_options=array()) {

        $tr_class = 'table_row';
        $tr_attr  = '';

        foreach( $tr_options as $opt => $val ) {
            switch( $opt ) {
                case 'class' : $tr_class .= " {$val}"; break;
                default:       $tr_attr  .= " {$opt}=\"{$val}\""; break;
            }
        }
        $html = <<<HERE
    <tr class="{$tr_class}" {$tr_attr} >

HERE;

        $td_class = $end_of_group ? 'table_cell' : 'table_cell_within_group';

        for( $col=0; $col < count($values); $col++ ) {

            $v = $values[$col];

            $align  = '';
            $valign = '';

            if( array_key_exists( $col, $td_options )) {

                $opt = $td_options[$col];

                if (array_key_exists( 'align',  $opt)) $align  = 'align="'. $opt['align']. '"';
                if (array_key_exists( 'valign', $opt)) $valign = 'valign="'.$opt['valign'].'"';
            }

            // Replace empty values with non-breakable space to avoid screwing up CSS
            // for table cells on Mozilla Firefox & IE browsers.
            //
            if( $v == '' ) $v = '&nbsp;';
            $html .= <<<HERE
      <td class="{$td_class}" {$align} {$valign}>{$v}</td>

HERE;
        }
        $html .= <<<HERE
    </tr>

HERE;
        return $html;
    }

    static function table_end_html() {
        return <<<HERE
  </tbody>
</table>

HERE;
    }

    static function decorated_experiment_contact_info( $experiment ) {
        return preg_replace(
               '/(.*)[( ](.+@.+)[) ](.*)/',
               '<a class="link" href="javascript:show_email('."'$1','$2'".')" title="click to see e-mail address">$1</a>$3',
               $experiment->contact_info());
    }
    static function experiment_contacts( $experiment ) {
        $addresses = array();
        preg_match_all (
               '/[(]+(.+@.+)[)]+/',
               $experiment->contact_info(),
               $addresses);
        return $addresses[1];
    }
    static function experiment_contact_info( $experiment ) {
        
        $addresses = array();
        preg_match_all (
               '/[, ]?(.+)[( ]+(.+@.+)[) ]+/',
               $experiment->contact_info(),
               $addresses);

        $result = array() ;
        for ($i = 0; $i < count($addresses[0]); $i++) {

            $gecos = $addresses[1][$i] ;
            $email = $addresses[2][$i] ;

            // Extract UID from email if this is a SLAC user
            
            $idx = strpos($email, '@slac.stanford.edu') ;
            $uid = $idx ? substr($email, 0, $idx) : $addresses[1][$i] ;

            array_push($result, array (
                'gecos' => $gecos ,
                'email' => $email ,
                'uid'   => $uid)) ;
        }
        return $result;
    }
    static function decorated_experiment_status( $experiment ) {
        $status = $experiment->in_interval( LusiTime::now());
        if     ( $status > 0 ) return '<b><em style="color:gray">completed</em></b>';
        else if( $status < 0 ) return '<b><em style="color:green">in prep</em></b>';
        return '<b><em style="color:red">on-going</em></b>';
    }

    static function decorated_experiment_status_UP( $experiment ) {
        $status = $experiment->in_interval( LusiTime::now());
        if     ( $status > 0 ) return '<b><em style="color:gray">COMPLETED</em></b>';
        else if( $status < 0 ) return '<b><em style="color:green">IN PREPARATION</em></b>';
        return '<b><em style="color:red">ON-GOING</em></b>';
    }

    static function error_message( $msg ) {
        header( 'Content-type: text/plain' );
        header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
        header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
        echo '<span style="font-size:24px; font-weight:bold; color:red;">ERROR: '.$msg.'</span>';
    }
}
?>
