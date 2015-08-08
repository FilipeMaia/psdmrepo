<?php

# Needed to supress complains in the server's log files
date_default_timezone_set('America/Los_Angeles') ;

require_once 'dataportal/dataportal.inc.php' ;
require_once 'websrt/websrt.inc.php' ;

use DataPortal\DataPortal ;

use websrt\WebSrt ;

function notes2html ($release) {
    $html = '' ;
    foreach (explode("\n", $release->notes()) as $l) {
        $html .= ($html == '' ? '' : '<br>').$l ;
    }        
    return $html ;
}


try {

    $releases = WebSrt::releases() ;

?>



<!------------------- Document Begins Here ------------------------->
<?php
    DataPortal::begin( "Status of the Offline Releases" );
?>



<!------------------- Page-specific Styles ------------------------->

<link type="text/css" href="../webfwk/css/Table.css" rel="Stylesheet" />

<style type="text/css"> 

table {
  border-spacing: 2px;
}

div.compare-info {
  margin-top: 5px;
  margin-bottom: 20px;
  padding-left: 3px;
  max-width: 640px;
}
div.compare-info p {
  margin: 0px;
  padding: 0px;
}

tr.table_row:hover {
  cursor: pointer;
  background-color: aliceblue;
}
tr.table_row.select {
  background-color: #A6C9E2;
}

span.ui-icon-close.close:hover {
  cursor: pointer;
}

div.table_tags {
  float: left;
  margin-right:20px;
}

div.table_notes {
  float: left;
}



</style>
<!----------------------------------------------------------------->

<script type="text/javascript" src="/underscore/underscore-min.js"></script>

<script type="text/javascript" src="../webfwk/js/Table.js"></script>


<?php
    DataPortal::scripts( "page_specific_init" );
?>


<!------------------ Page-specific JavaScript ---------------------->


<script type="text/javascript">

var release_types = ['ana', 'data', 'web'] ;
var tabs = null ;

function page_specific_init () {

    tabs = $('#tabs').tabs() ;
    for (var i in release_types) {

        var cont = tabs.find('div#'+release_types[i]) ;
        cont.find('.table_row').click(function () {

            var tr = $(this) ;

            // -- check if the same row has been deselected

            if (tr.hasClass('select')) {
                tr.removeClass('select') ;
                return ;
            } else {
                tr.addClass('select') ;
            }

            // -- if that's another one then figure out which releases need to be compared
            //    and create the new panel
         
            var reltype = tr.attr('reltype') ;

            var cont = tabs.find('div#'+reltype) ;
            var tr_select = cont.find('.table_row.select') ;
            if (tr_select.length > 1) {

                tr_select.removeClass('select') ;

                var releases = [] ;
                tr_select.each(function () {
                    var tr = $(this) ;
                    var relname = tr.attr('relname') ;
                    releases.push(relname) ;
                }) ;
                compare_releases(releases) ;
            }
        }) ;
    }
}

var next_tab_id = 0 ;

function compare_releases (releases) {

    // -- create a new tab panel to show differences between releases
 
    var panelId = 'compare_'+(next_tab_id++) ;
    var name = releases[0]+' - '+releases[1] ;
 
    var li_html =
'<li><a href="#'+panelId+'" >'+name+'</a> <span class="ui-icon ui-icon-close close"></span></li>' ;

    var div_html =
'<div id="'+panelId+'">' +
'  <div class="tab-inline-content">' +
'    <div class="table_tags">Loading...</div>' +
'    <div class="table_notes">Loading...</div>' +
'    <div style="clear:both;"></div>' +
'  </div>' +
'</div>' ;

    tabs.children('.ui-tabs-nav').append (li_html) ;
    tabs.append(div_html) ;
    tabs.tabs("refresh") ;

    // -- move the focus to the new tab

    var panel2activate = tabs.children('.ui-tabs-panel').length - 1 ;
    tabs.tabs("select", panel2activate) ;
    tabs.tabs("option", "active", panel2activate) ;

    // -- set up the event handler for removing the panel  

    tabs.find('span.ui-icon-close.close').live('click', function () {
        var panelId = $(this).closest('li').attr('aria-controls') ;
        tabs.find('a[href="#'+panelId+'"]').closest('li').remove() ;
        tabs.find($('div#'+panelId)).remove() ;
        tabs.tabs('refresh') ;
    }) ;
    var cont = tabs.children('div#'+panelId).children('div.tab-inline-content') ;
    load_release_diff(releases, cont) ;
}

function load_release_diff (release_names, cont) {
    var params = {
        names: JSON.stringify(release_names)
    } ;
    web_service_POST (
        "../websrt/ws/release_diff_get.php" ,
        params ,
        function (data) {
            display_release_diff(data, cont) ;
        }
    ) ;
}

function web_service_POST (url, params, on_success, on_failure) {
    var jqXHR = $.post(url, params, function (data) {
        if (data.status != 'success') {
            if (on_failure) on_failure(data.message) ;
            else            report_error('Web Service Request Failed', data.message) ;
            return ;
        }
        if (on_success) on_success(merge_tags(data)) ;
    },
    'JSON').error(function () {
        var message = 'Web service request to '+url+' failed because of: '+jqXHR.statusText ;
        if (on_failure) on_failure(message) ;
        else            report_error(message, null) ;
    }) ;
} ;

function display_release_diff (release_data, cont) {

    var base_url = 'http://java.freehep.org/svn/repos/psdm/list/' ;

    // -- table of tags

    var t_tags_hdr = [{name: 'Package'}] ;
    for (var i in release_data.names) {
        var r_name = release_data.names[i] ;
        t_tags_hdr.push({name: r_name, sorted: false}) ;
    }
    if (release_data.names.length >= 2) {
        t_tags_hdr.push({name: 'Changes'}) ;
    }
    var t_tags_rows = [] ;
    for (var i in release_data.packages) {
        var pkg = release_data.packages[i] ;
        var row = [pkg] ;
        for (var j in release_data.names) {
            var r_name = release_data.names[j] ;
            var tags = release_data.tags[r_name] ;
            if (pkg in tags) {
                var tag = tags[pkg] ;
                row.push('<a class="link" href="'+base_url+pkg+'/tags/'+tag+'" target="_blank">'+tag+'</a>') ;
            } else {
                row.push('&nbsp;') ;
            }
        }

        // -- highlight differences (only if 2 or more releases are provided)

        if (release_data.names.length >= 2) {
            var diff_tags = false ;
            for (var k = 2; k < row.length; k++) {
                if (row[k-1] !== row[k]) {
                    diff_tags = true ;
                    break ;
                } 
            }
            if (diff_tags) {
                row.push('<div style="background-color:red; width:8px; height:8px;">&nbsp;</div>') ;
            } else {
                row.push('&nbsp;') ;
            }
        }
        t_tags_rows.push(row) ;
    }
    var t_tags_table = new Table (
        cont.children('div.table_tags') ,
        t_tags_hdr ,
        t_tags_rows
    ) ;
    t_tags_table.display() ;

    // -- table of changes between releases

    var t_notes_hdr = [
        {name: 'Release', sorted: false} ,
        {name: 'Notes',   sorted: false}
    ] ;

    var t_notes_rows = [] ;
    for (var i in release_data.notes) {
        var e = release_data.notes[i] ;
        t_notes_rows.push([
            e.release ,
            e.notes
        ]) ;
    }

    var t_notes_table = new Table (
        cont.children('div.table_notes') ,
        t_notes_hdr ,
        t_notes_rows
    ) ;
    t_notes_table.display() ;
}

function merge_tags (data) {
    var release_data = {
        names:    [] ,  // -- original order of release names preserved
        packages: [] ,  // -- package names sorted alphabetically
        tags:     {} ,  // -- the dictionary of tags for each release
        notes:    []    // -- release notes
    } ;
 
    var packages = {} ;     // -- unique package names which were found accross
                            //    releases

    for (var i in data.releases) {
        var release = data.releases[i] ;
        var r_name = release.name ;
        release_data.names.push(r_name) ;
        release_data.tags[r_name] = release.tags ;
        for (var pkg in release.tags) {
            packages[pkg] = true ;
        }
    }
    for (var i in data.notes) {
        var e = data.notes[i] ;
        release_data.notes.push({
            release: e.release ,
            notes: _.reduce (
                e.notes.split('\n') ,
                function (html, line) {
                    var result = html + line + '<br>' ;
                    return result ;
                } ,
                '')
        }) ;
    }
    for (var pkg in packages) {
        release_data.packages.push(pkg) ;
    }
    release_data.packages.sort() ;

    return release_data ;
}

</script>
<!----------------------------------------------------------------->


<?php
    DataPortal::body( "Software Release Tools: Releases" );
?>




<!------------------ Page-specific Document Body ------------------->
<?php

    function table_header () {
        $row = array (
            array ('name' => 'Release',        'width' =>  80) ,
            array ('name' => 'Built',          'width' =>  70) ,
            array ('name' => 'Available',      'width' =>  20) ,
            array ('name' => '# new pkg',      'width' =>  80) ,
            array ('name' => '# removed pkg',  'width' =>  80) ,
            array ('name' => '# modified pkg', 'width' =>  80) ,
            array ('name' => 'Notes')
        ) ;
        return DataPortal::table_begin_html($row) ;
    }

    $td_opt = array (
        array('valign' => 'top') ,
        array('valign' => 'top') ,
        array('valign' => 'top') ,
        array('valign' => 'top') ,
        array('valign' => 'top') ,
        array('valign' => 'top') ,
        array('valign' => 'top')
    ) ;

    function table_row ($r) {
        global $td_opt ;
        $row = array (
            $r->name() ,
            $r->deployed_date() ,
            $r->on_disk() ? 'Yes' : '' ,
            $r->num_new_pkg() ? $r->num_new_pkg() : '' ,
            $r->num_removed_pkg() ? $r->num_removed_pkg() : '' ,
            $r->num_modified_pkg() ? $r->num_modified_pkg() : '' ,
            notes2html($r)
        ) ;
        $end_of_group = true ;
        $tr_opt = array (
            'reltype' => $r->type() ,
            'relname' => $r->name()
        ) ;
        return DataPortal::table_row_html($row, $end_of_group, $td_opt, $tr_opt) ;
    }

    $release_types = array (
        array (
            'name'     => 'Analysis' ,
            'id'       => 'ana' ,
            'releases' => $releases['ana'] ) ,
        array (
            'name'     => 'Data Management (legacy)' ,
            'id'       => 'data' ,
            'releases' => $releases['data'] ) ,
        array (
            'name'     => 'Data Management' ,
            'id'       => 'dm' ,
            'releases' => $releases['dm'] ) ,
        array (
            'name'     => 'Web' ,
            'id'       => 'web' ,
            'releases' => $releases['web'] )) ;

    $tabs = array() ;

    foreach ($release_types as $t) {
        $reltype = $t['id'] ;
        $html =<<<HERE
<div class="compare-info">
  <p><b>VIEWING/COMPARING RELEASES:</b>
     Select two releases by clicking on the corresponding rows. Selection of the
     second release will automatically initiate the release vieweing/comparision dialog
     open in a new tab.
  </p>
</div>
HERE;
        $html .= table_header() ;
        foreach ($t['releases'] as $r) {
            $html .= table_row($r) ;
        }
        $html .= DataPortal::table_end_html() ;

        array_push (
            $tabs ,
            array (
                'name'  => $t['name'] ,
                'id'    => $t['id'] ,
                'html'  => $html ,
                'class' => 'tab-inline-content'
            )
        ) ;
    }

    /* Print the whole tab and its contents (including sub-tabs if any).
     */
    DataPortal::tabs("tabs", $tabs) ;
?>
<!----------------------------------------------------------------->






<?php
    DataPortal::end() ;
?>
<!--------------------- Document End Here -------------------------->

<?php

} catch (Exception $e) { print '<pre>'.$e.'</pre>' ; }


?>     