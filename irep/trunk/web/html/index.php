<?php

require_once 'authdb/authdb.inc.php' ;
require_once 'irep/irep.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use AuthDB\AuthDB ;
use AuthDB\AuthDBException ;

use Irep\Irep ;
use Irep\IrepException ;

use LusiTime\LusiTimeException ;


$document_title = 'PCDS Inventory And Repair Database:' ;
$document_subtitle = 'Electronic Equipment' ;

$required_field_html = '<span style="color:red ; font-size:110% ; font-weight:bold ;"> * </span>' ;

try {

    $authdb = AuthDB::instance() ;
    $authdb->begin() ;

    $irep = Irep::instance() ;
    $irep->begin() ;

?>


<!-- Document Begins Here -->


<!DOCTYPE html>
<html>

<head>
<title><?php echo $document_title ?></title>
<meta http-equiv="Content-Type" content="text/html ; charset=ISO-8859-1">

<link type="text/css" href="/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css" rel="Stylesheet" />

<link type="text/css" href="css/fwk.css" rel="Stylesheet" />
<link type="text/css" href="css/irep.css" rel="Stylesheet" />

<link type="text/css" href="../portal/css/Table.css" rel="Stylesheet" />

<script type="text/javascript" src="/jquery/js/jquery-1.8.2.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui-1.9.1.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.printElement.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.json.js"></script>

<script type="text/javascript" src="/underscore/underscore-min.js"></script>

<script type="text/javascript" src="js/datetime.js"></script>
<script type="text/javascript" src="js/fwk.js"></script>
<script type="text/javascript" src="js/ws.js"></script>
<script type="text/javascript" src="js/equipment.js"></script>
<script type="text/javascript" src="js/issues.js"></script>
<script type="text/javascript" src="js/dictionary.js"></script>
<script type="text/javascript" src="js/admin.js"></script>

<script type="text/javascript" src="../portal/js/config.js"></script>
<script type="text/javascript" src="../portal/js/Table.js"></script>

<!-- Window layout styles and support actions -->

<style type="text/css">

#equipment-inventory-controls {
  margin-bottom: 8px;
  padding-left: 10px;
}
#equipment-inventory-controls-left {
  margin-right: 20px;
}

#equipment-inventory-info {
  color: maroon;
}

.form-elem {
  width:100%;
}

div.equipment-grid-cell {
  margin-right: 20px;
  margin-bottom: 20px;

  background-color: #f4f4f4;
  /*
  background-color: #f0f0d0;
  */
  border: 1px solid #c8c8c8;

  border-radius: 6px;
  -moz-border-radius: 6px;
  /*
  border-bottom-left-radius: 0;
  border-bottom-right-radius: 0;
  -moz-border-radius-bottomleft: 0;
  -moz-border-radius-bottomright: 0;
  */
}
div.equipment-grid-cell div.header {
  padding: 2px;
  background-color: #c0c0c0;
}

div.equipment-grid-cell div.body {
}
div.equipment-grid-cell div.footer {
  background-color: #c0c0c0;
}

div.equipment-grid-cell pre {
  margin: 0px;
}
td.equipment-edit-cell {
  border: 0;
  padding-right: 20px;
}
div.equipment-tag-hdr {
  margin-left: 5px;
  margin-right: 5px;
  font-weight: bold;
}
div.equipment-tag {
  margin-right: 5px;
  font-style: italic;
}
div.equipment-attachment-edit-entry {
  padding: 10px;
}
div.equipment-attachment-new-edit-entry {
  padding: 10px;
}
div.equipment-tag-edit-entry {
  padding: 10px;
}
div.equipment-tag-new-edit-entry {
  padding: 10px;
}
div.equipment-edit-entry-modified {
  background-color: #ffdcdc;
}
span.toggler {
  background-color: #ffffff;
  border: 1px solid #c0c0c0;
  border-radius: 4px;
  -moz-border-radius: 4px;
  cursor: pointer;
}

div.visible {
  display: block;
}

div.hidden {
  display: none;
}

button.visible {
  display: block;
}

button.hidden {
  display: none;
}

span.form_element_info {
  color: maroon;
}
</style>


<script type="text/javascript">

/* ----------------------------------------------------------
 *             APPLICATION-SPECIFIC INITIALIZATION
 * ----------------------------------------------------------
 */
var config = new config_create('irep') ;

var global_current_user = {
    uid:               '<?php echo $authdb->authName        () ;         ?>' ,
    is_other:           <?php echo $irep->is_other          ()?'1':'0' ; ?>  ,
    is_administrator:   <?php echo $irep->is_administrator  ()?'1':'0' ; ?>  ,
    can_edit_inventory: <?php echo $irep->can_edit_inventory()?'1':'0' ; ?>  ,
    has_dict_priv:      <?php echo $irep->has_dict_priv     ()?'1':'0' ; ?>
} ;

var global_users   = [] ;
var global_editors = [] ;
<?php
    foreach ($irep->users() as $user) {
        echo "global_users.push('{$user->uid()}') ;\n" ;
        if ($user->is_administrator() || $user->is_editor()) echo "global_editors.push('{$user->uid()}') ;\n" ;
    }
?>
function global_get_editors() {
    var editors = admin.editors() ;
    if (editors) return editors ;
    return global_editors ;
}
var applications = {
    'p-appl-equipment'  : equipment ,
    'p-appl-issues'     : issues ,
    'p-appl-dictionary' : dict ,
    'p-appl-admin'      : admin
} ;

var current_application = null ;

var select_app         = 'equipment' ;
var select_app_context = 'inventory' ;
<?php
$known_apps = array(
    'equipment'  => True ,
    'issues'     => True ,
    'dictionary' => True ,
    'admin'      => True
) ;
if (isset($_GET['app'])) {
    $app_path = explode(':', strtolower(trim($_GET['app']))) ;
    $app = $app_path[0] ;
    if (array_key_exists($app, $known_apps)) {
        echo "select_app = '{$app}' ;" ;
        echo "select_app_context = '".(count($app_path) > 1 ? $app_path[1] : "")."' ;" ;
    }
}
?>
var select_params = {
    equipment_id: null
} ;
<?php
if (isset($_GET['equipment_id'])) {
    $equipment_id = intval(trim($_GET['equipment_id'])) ;
    if ($equipment_id) echo "select_params.equipment_id = {$equipment_id} ;" ;
}
?>
/* Event handler for application selections from the top-level menu bar:
 * - fill set the current application context.
 */
function m_item_selected(item) {

    if (current_application == applications[item.id]) return ;
    if ((current_application != null) && (current_application != applications[item.id])) {
        current_application.if_ready2giveup(function() {
            m_item_selected_impl(item) ;
        }) ;
        return ;
    }
    m_item_selected_impl(item) ;
}

function m_item_selected_impl(item) {

    current_application = applications[item.id] ;

    $('.m-select').removeClass('m-select') ;
    $(item).addClass('m-select') ;
    $('#p-left > #v-menu .visible').removeClass('visible').addClass('hidden') ;
    $('#p-left > #v-menu > #'+current_application.name).removeClass('hidden').addClass('visible') ;

    $('#p-center .application-workarea.visible').removeClass('visible').addClass('hidden') ;
    var wa_id = current_application.name ;
    if (current_application.context != '') wa_id += '-'+current_application.context ;
    $('#p-center .application-workarea#'+wa_id).removeClass('hidden').addClass('visible') ;

    current_application.select_default() ;
    v_item_selected($('#v-menu > #'+current_application.name).children('.v-item#'+current_application.context)) ;
    
    set_context(current_application) ;
}

/* Event handler for vertical menu item (actual commands) selections:
 * - dim the poreviously active item
 * - hightlight the new item
 * - change the current context
 * - execute the commands
 * - switch the work area (make the old one invisible, and the new one visible)
 */
function v_item_selected(item) {

     var item = $(item) ;
    if ($(item).hasClass('v-select')) return ;

    if (current_application.context != item.attr('id')) {
        current_application.if_ready2giveup(function() {
            v_item_selected_impl(item) ;
        }) ;
        return ;
    }
    v_item_selected_impl(item) ;
}

function v_item_selected_impl(item) {

    $('#'+current_application.name).find('.v-item.v-select').each(function(){
        $(this).children('.ui-icon').removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e') ;
        $(this).removeClass('v-select') ;
    }) ;

    $(item).children('.ui-icon').removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s') ;
    $(item).addClass('v-select') ;

    /* Hide the older work area
     */
    var wa_id = current_application.name ;
    if (current_application.context != '') wa_id += '-'+current_application.context ;
    $('#p-center > #application-workarea > #'+wa_id).removeClass('visible').addClass('hidden') ;

    current_application.select(item.attr('id')) ;

    /* display the new work area
     */
    wa_id = current_application.name ;
    if (current_application.context != '') wa_id += '-'+current_application.context ;
    $('#p-center > #application-workarea > #'+wa_id).removeClass('hidden').addClass('visible') ;

    set_context(current_application) ;
}

$(function() {

    $('.m-item').click(function() { m_item_selected (this) ; }) ;
    $('.v-item').click(function() { v_item_selected (this) ; }) ;

    $('#p-search-text').keyup(function(e) { if (($(this).val() != '') && (e.keyCode == 13)) global_simple_search() ; }) ;

    // Make sure the dictionaries are loaded
    //
    dict.init() ;
    admin.init() ;

    // Finally, activate the selected application.
    //
    for (var id in applications) {
        var application = applications[id] ;
        if (application.name == select_app) {
            $('#p-menu').children('#p-appl-'+select_app).each(function() { m_item_selected(this) ; }) ;
            if ('' == select_app_context) {
                v_item_selected($('#v-menu > #'+select_app+' > #'+application.default_context)) ;
                application.select_default() ;
            } else {
                v_item_selected($('#v-menu > #'+select_app+' > #'+select_app_context)) ;
                application.select(select_app_context) ;            
            }
            switch(application.name) {
            case 'equipment':
                switch(application.context) {
                case 'inventory':
                    if (select_params.equipment_id) global_search_equipment_by_id(select_params.equipment_id) ;
                    break ;
                }
                break ;
            }
        }
    }
}) ;

/* ------------------------------------------------------
 *             CROSS_APPLICATION EVENT HANDLERS
 * ------------------------------------------------------
 */
function global_switch_context(application_name, context_name) {
    for (var id in applications) {
        var application = applications[id] ;
        if (application.name == application_name) {
            $('#p-menu').children('#'+id).each(function() {    m_item_selected(this) ; }) ;
            v_item_selected($('#v-menu > #'+application_name).children('.v-item#'+context_name)) ;
            if (context_name != null) application.select(context_name) ;
            else application.select_default() ;
            return application ;
        }
    }
    return null ;
}
function global_simple_search                    ()   { global_switch_context('equipment', 'inventory').simple_search($('#p-search-text').val()) ; }
function global_search_equipment_by_id           (id) { global_switch_context('equipment', 'inventory').search_equipment_by(id) ; }
function global_search_equipment_by_location     (id) { global_switch_context('equipment', 'inventory').search_equipment_by_location(id) ; }
function global_search_equipment_by_room         (id) { global_switch_context('equipment', 'inventory').search_equipment_by_room(id) ; }
function global_search_equipment_by_manufacturer (id) { global_switch_context('equipment', 'inventory').search_equipment_by_manufacturer(id) ; }
function global_search_equipment_by_model        (id) { global_switch_context('equipment', 'inventory').search_equipment_by_model(id) ; }
function global_search_equipment_by_slacid_range (id) { global_switch_context('equipment', 'inventory').search_equipment_by_slacid_range(id) ; }
function global_search_equipment_by_status       (id) { global_switch_context('equipment', 'inventory').search_equipment_by_status(id) ; }
function global_search_equipment_by_status2      (id) { global_switch_context('equipment', 'inventory').search_equipment_by_status2(id) ; }

function global_export_equipment(search_params,outformat) {
    search_params.format = outformat ;
    var html = '<img src="../logbook/images/ajaxloader.gif" />' ;
    var dialog = report_action('Generating Document: '+outformat,html) ;
    var jqXHR = $.get(
        '../irep/ws/equipment_inventory_search.php', search_params,
        function(data) {
            if (data.status != 'success') {
                report_error(data.message) ;
                dialog.dialog('close') ;
                return ;
            }
            var html = 'Document is ready to be downloaded from this location: <a class="link" href="'+data.url+'" target="_blank" >'+data.name+'</a>' ;
            dialog.html(html) ;
        },
        'JSON'
    ).error(
        function () {
            report_error('failed because of: '+jqXHR.statusText) ;
            dialog.dialog('close') ;
        }
    ).complete(
        function () {
        }
    ) ;
}

function global_equipment_status2rank(status) {
    switch(status) {
        case 'Unknown': return 0 ;
    }
    return -1 ;
}
function global_equipment_sorter_by_status       (a,b) { return global_equipment_status2rank(a.status) - global_equipment_status2rank(b.status) ; }
function sort_as_text                            (a,b) { return a == b ? 0 : (a < b ? -1 : 1) ; }
function global_equipment_sorter_by_manufacturer (a,b) { return sort_as_text(a.manufacturer, b.manufacturer) ; }
function global_equipment_sorter_by_model        (a,b) { return sort_as_text(a.model,        b.model) ; }
function global_equipment_sorter_by_location     (a,b) { return sort_as_text(a.location,     b.location) ; }
function global_equipment_sorter_by_modified     (a,b) { return a.modified.time_64 - b.modified.time_64 ; }

</script>

</head>

<body onresize="resize()">

<div id="p-top">
  <div id="p-top-header">
    <div id="p-top-title">
      <div style="float:left ; padding-left:15px ; padding-top:10px ;">
        <span id="p-title"><?php echo $document_title?></span>
        <span id="p-subtitle"><?php echo $document_subtitle?></span>
      </div>
      <div id="p-login" style="float:right ;" >
        <div style="float:left ; padding-top:20px ;" class="not4print" >
          <a href="javascript:printer_friendly()" title="Printer friendly version of this page"><img src="img/PRINTER_icon.gif" style="border-radius: 5px ;" /></a>
        </div>
        <div style="float:left ; margin-left:10px ;" >
          <table><tbody>
            <tr>
              <td>&nbsp;</td>
              <td>[<a href="javascript:logout()" title="close the current WebAuth session">logout</a>]</td></tr>
            <tr>
              <td>User:&nbsp;</td>
              <td><b><?php echo $authdb->authName()?></b></td></tr>
            <tr>
              <td>Session expires in:&nbsp;</td>
              <td id="auth_expiration_info"><b>00:00.00</b></td></tr>
          </tbody></table>
        </div>
        <div style="clear:both ;" class="not4print"></div>
      </div>
      <div style="clear:both ;"></div>
    </div>
    <div id="p-menu">
      <div class="m-item m-item-first m-select" id="p-appl-equipment" >Equipment</div>
      <div class="m-item m-item-next"           id="p-appl-issues"    >Issues</div>
      <div class="m-item m-item-next"           id="p-appl-dictionary">Dictionary</div>
      <div class="m-item m-item-last"           id="p-appl-admin"     >Admin</div>
      <div class="m-item-end"></div>
    </div>
    <div id="p-context-header">
      <div id="p-context" style="float:left"></div>
      <div id="p-search" style="float:right">
        quick search: <input type="text" id="p-search-text" value="" size=16 title="enter full or partial attribute of equipment to search, then press RETURN to proceed"  style="font-size:80% ; padding:1px ; margin-top:6px ;" />
      </div>
      <div style="clear:both ;"></div>
    </div>
  </div>
</div>

<div id="p-left">

<div id="v-menu">

    <div id="menu-title"></div>

    <div id="equipment" class="visible">
      <div class="v-item" id="inventory">
        <div class="ui-icon ui-icon-triangle-1-s" style="float:left ;"></div>
        <div style="float:left ;" >Inventory</div>
        <div style="clear:both ;"></div>
      </div>
      <div class="v-item" id="add">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left ;"></div>
        <div class="link" style="float:left ;" >Add New Equipment</div>
        <div style="clear:both ;"></div>
      </div>
    </div>

    <div id="issues" class="visible">
      <div class="v-item" id="search">
        <div class="ui-icon ui-icon-triangle-1-s" style="float:left ;"></div>
        <div style="float:left ;" >Search</div>
        <div style="clear:both ;"></div>
      </div>
      <div class="v-item" id="reports">
        <div class="ui-icon ui-icon-triangle-1-s" style="float:left ;"></div>
        <div style="float:left ;" >Reports</div>
        <div style="clear:both ;"></div>
      </div>
    </div>

    <div id="dictionary" class="hidden">
      <div class="v-item" id="manufacturers">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left ;"></div>
        <div style="float:left ;" >Manufacturers/Models</div>
        <div style="clear:both ;"></div>
      </div>
      <div class="v-item" id="locations">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left ;"></div>
        <div style="float:left ;" >Locations</div>
        <div style="clear:both ;"></div>
      </div>
      <div class="v-item" id="statuses">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left ;"></div>
        <div style="float:left ;" >Statuses</div>
        <div style="clear:both ;"></div>
      </div>
    </div>

    <div id="admin" class="hidden">
      <div class="v-item" id="access">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left ;"></div>
        <div style="float:left ;" >Access Control</div>
        <div style="clear:both ;"></div>
      </div>
      <div class="v-item" id="notifications">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left ;"></div>
        <div style="float:left ;" >E-mail Notifications</div>
        <div style="clear:both ;"></div>
      </div>
      <div class="v-item" id="slacid">
        <div class="ui-icon ui-icon-triangle-1-e" style="float:left ;"></div>
        <div style="float:left ;" >SLACid Numbers</div>
        <div style="clear:both ;"></div>
      </div>
    </div>

  </div>
</div>

<div id="p-splitter"></div>

<div id="p-center">
  <div id="application-workarea">

    <!-- Inventory of all known equipment -->
    <div id="equipment-inventory" class="application-workarea hidden">
      <div id="equipment-inventory-controls">
        <div id="equipment-inventory-controls-left" style="float:left ;">
          <form id="equipment-inventory-form">
            <table><tbody>
              <tr>
                <td class="form_table_key" >Manufacturer  </td> <td class="form_table_val" ><select name="manufacturer" class="form-elem" ></select></td>
                <td class="form_table_key" >Custodian     </td> <td class="form_table_val" ><select name="custodian"    class="form-elem" ></select></td>
                <td class="form_table_key" >Serial #      </td> <td class="form_table_val" ><input  name="serial"       class="form-elem" type="text" size="4" value="" ></td>
                <td class="form_table_key" >Status        </td> <td class="form_table_val" ><select name="status"       class="form-elem" ></select></td>
                <td class="form_table_key" >Tag           </td> <td class="form_table_val" ><select name="tag"          class="form-elem" ></select></td>
              </tr>
              <tr>
                <td class="form_table_key" >Model         </td> <td class="form_table_val" ><select name="model"        class="form-elem" ></select></td>
                <td class="form_table_key" >Location      </td> <td class="form_table_val" ><select name="location"     class="form-elem" ></select></td>
                <td class="form_table_key" >PC #          </td> <td class="form_table_val" ><input  name="pc"           class="form-elem" type="text" size="4" value="" title="property control number" /></td>
                <td class="form_table_key" >Sub-status    </td> <td class="form_table_val" ><select name="status2"      class="form-elem" ></select></td>
              </tr>
              <tr>
                <td class="form_table_key" >&nbsp;        </td> <td class="form_table_val" >&nbsp;</td>
                <td class="form_table_key" >Room          </td> <td class="form_table_val" ><select name="room"         class="form-elem" ></select></td>
                <td class="form_table_key" >SLAC ID #     </td> <td class="form_table_val" ><input  name="slacid"       class="form-elem" type="text" size="4" value="" ></td>
              </tr>
              <tr>
                <td class="form_table_key" >Description   </td> <td class="form_table_val" colspan="3" ><input  name="description"  class="form-elem" type="text" size="4" value="" title="search in the model descriptions"/></td>
                <td class="form_table_key" >Notes         </td> <td class="form_table_val" colspan="3" ><input name="notes"         class="form-elem" type="text" size="10" value="" title="search in the equipment notes"/></td>
              </tr>
            </tbody></table>
          </form>
        </div>
        <div style="float:left ; margin-left:20px ;">
          <button name="search" title="refresh the list">Search</button>
          <button name="reset"  title="reset the search form to the default state">Reset Form</button>
        </div>
        <div style="clear:both ;"></div>
      </div>
      <div style="float:right ;" id="equipment-inventory-info">[ Last updated: ]</div>
      <div style="clear:both ;"></div>

      <div id="tabs" style="font-size:12px;">
        <ul>
          <li><a href="#results">Search Results</a></li>
        </ul>

        <div id="results" >
          <div style=" border:solid 1px #b0b0b0; padding:20px;" >
            <div style="float:left;">
              <button class="export" name="excel" title="Export into Microsoft Excel 2007 File"><img src="../irep/img/EXCEL_icon.gif" /></button>
            </div>
            <div style="float:left; margin-left:20px;">
              <center><b>&nbsp;</b></center>
              <div id="view">
                <input type="radio" id="view_table" name="view" checked="checked" ><label for="view_table" title="view as a table" ><img src="../irep/img/table.png" /></label>
                <input type="radio" id="view_grid"  name="view"                   ><label for="view_grid"  title="view as a grid"  ><img src="../irep/img/stock_table_borders.png" /></label>
              </div>
            </div>
            <div style="float:left; margin-left:20px;">
                <center><b>&nbsp;</b></center>
                <input type="checkbox" id="option_model_image"        ><label for="option_model_image"       > display images of models</label><br>
                <input type="checkbox" id="option_attachment_preview" ><label for="option_attachment_preview"> preview attachments     </label>
            </div>
            <div style="float:left; margin-left:20px;">
                <center><b>&nbsp;</b></center>
                <input type="checkbox" id="option_model_descr"        ><label for="option_model_descr"       > display descriptions of models</label><br>
            </div>
            <div style="clear:both ;"></div>
            <div id="equipment-inventory-table" style="margin-top:10px;" ></div>
          </div>
        </div>
      </div>
    </div>

    <!-- New equipment registration dialog -->
    <div id="equipment-add" class="application-workarea hidden">
<?php
    if ($irep->can_edit_inventory()) {
?>
      <div style="margin-bottom:20px ; border-bottom:1px dashed #c0c0c0 ;">
        <div style="float:left ;">
          <div style="margin-bottom:10px ; width:480px ;">
            When making a clone of an existing equipment record make sure the  serial number, Property Control (PC) number,
            and a SLAC ID of the new equipment differ from the original one. All other attributes of the original equipment
            will be copied into the new one. The copied equipment will all be put into the 'Unknown' state.
          </div>
          <form id="equipment-add-form">
            <table><tbody>
              <tr><td><b>Manufacturer:<?php echo $required_field_html; ?></b></td>
                  <td colspan="3"><select name="manufacturer" class="equipment-add-form-element" ></select>
                      <span class="form_element_info"></span></td>
              </tr>
              <tr><td><b>Model:<?php echo $required_field_html; ?></b></td>
                  <td colspan="3"><select name="model" class="equipment-add-form-element" ></select>
                      <span class="form_element_info"></span></td>
              </tr>
              <tr><td><b>Serial number:</b></td>
                  <td><input type="text" name="serial" class="equipment-add-form-element" size="20" style="padding:2px ;" value="" /></td>
              </tr>
              <tr><td><b>Property Control #:</b></td>
                  <td><input type="text" name="pc"  size="20" style="padding:2px ;" value="" /></td></tr>
              <tr><td><b>SLAC ID:<?php echo $required_field_html; ?></b></td>
                  <td colspan="3"><input type="text" name="slacid" class="equipment-add-form-element" size="20" style="padding:2px ;" value="" />
                      <span class="form_element_info"></span></td>
              </tr>
              <tr><td><b>Location:</b></td>
                  <td><select name="location" class="equipment-add-form-element" ></select></td>
                  <td><b>Room:</b></td>
                  <td><select name="room" class="equipment-add-form-element" ></select></td>
              </tr>
              <tr><td colspan="2">&nbsp;</td>
                  <td><b>Rack:</b></td>
                  <td><input type="text" name="rack" class="equipment-add-form-element" size="20" value="" /></td>
              </tr>
              <tr><td colspan="2">&nbsp;</td>
                  <td><b>Elevation:</b></td>
                  <td><input type="text" name="elevation" class="equipment-add-form-element" size="20" value="" /></td>
              </tr>
              <tr><td><b>Custodian:</b></td>
                  <td colspan="3"><input type="text" name="custodian" size="20" style="padding:2px ;" value="" />
                  ( known custodians: <select name="custodian"></select> )</td>
              </tr>
              <tr><td><b>Notes: </b></td>
                  <td colspan="3"><textarea cols=54 rows=4 name="description" style="padding:4px ;" title="Here be arbitrary notes for this equipment"></textarea></td>
              </tr>
            </tbody></table>
          </form>
        </div>
        <div style="float:left ; padding:5px ;">
          <div>
            <button id="equipment-add-save">Create</button>
            <button id="equipment-add-reset">Reset Form</button>
          </div>
          <div style="margin-top:5px ;" id="equipment-add-info" >&nbsp;</div>
        </div>
        <div style="clear:both ;"></div>
      </div>
      <?php echo $required_field_html; ?> required field
<?php
    } else {
        $admin_access_href = "javascript:global_switch_context('admin','access')" ;
?>
      <br><br>
      <center>
        <span style="color: red ; font-size: 175% ; font-weight: bold ; font-family: Times, sans-serif ;">
          A c c e s s &nbsp; E r r o r
        </span>
      </center>
      <div style="margin: 10px 10% 10px 10% ; padding: 10px ; font-size: 125% ; font-family: Times, sans-serif ; border-top: 1px solid #b0b0b0 ;">
        We're sorry! Your SLAC UNIX account <b><?php echo $authdb->authName(); ?></b> has no sufficient permissions for this operation.
        Normally we assign this task to authorized <a href="<?php echo $admin_access_href; ?>">database editors</a>.
        Please contact administrators of this application if you think you need to add/edit equipment records.
        A list of administrators can be found in the <a href="<?php echo $admin_access_href; ?>">Access Control</a> section of the <a href="<?php echo $admin_access_href; ?>">Admin</a> tab of this application.
      </div>
<?php
    }
?>
    </div>

    <!-- Search equipment issues -->
    <div id="issues-search" class="application-workarea hidden">
      <p>This is still under implementation</p>
    </div>

    <!-- Produce reports on equipment issues -->
    <div id="issues-reports" class="application-workarea hidden">
      <p>Sorry, this feature is not implemented in this version of the software!</p>
    </div>

    <!-- Manufactures (dictionary) -->
    <div id="dictionary-manufacturers" class="application-workarea hidden">
      <div><button id="dictionary-manufacturers-reload" title="reload the dictionary from the database">Reload</button></div>
      <div style="float:left ;">
        <div style="margin-top:20px ;">
          <div style="float:left ; "><input type="text" size="12" name="manufacturer2add" title="fill in new manufacturer name, then press RETURN to save" /></div>
              <div style="float:left ; padding-top:4px ; color:maroon ;">  &larr; add new manufacturer here</div>
          <div style="clear:both ; "></div>
        </div>
        <div id="dictionary-manufacturers-manufacturers"></div>
      </div>
      <div style="float:left; margin-left:20px;">
        <div style="margin-top:20px ;">
          <div style="float:left ; "><input type="text" size="12" name="model2add" title="fill in new model name, then press RETURN to save" /></div>
          <div style="float:left ; padding-top:4px ; color:maroon ;">  &larr; add new model here</div>
          <div style="clear:both ; "></div>
        </div>
        <div id="dictionary-manufacturers-models"></div>
      </div>
      <div style="clear:both ; "></div>
    </div>

    <!-- Locations (dictionary) -->
    <div id="dictionary-locations" class="application-workarea hidden">
      <div><button id="dictionary-locations-reload" title="reload the dictionary from the database">Reload</button></div>
      <div style="float:left ;">
        <div style="margin-top:20px ;">
          <div style="float:left ; "><input type="text" size="12" name="location2add" title="fill in new location name, then press RETURN to save" /></div>
              <div style="float:left ; padding-top:4px ; color:maroon ;">  &larr; add new location here</div>
          <div style="clear:both ; "></div>
        </div>
        <div id="dictionary-locations-locations"></div>
      </div>
      <div style="float:left; margin-left:20px;">
        <div style="margin-top:20px ;">
          <div style="float:left ; "><input type="text" size="12" name="room2add" title="fill in new room name, then press RETURN to save" /></div>
              <div style="float:left ; padding-top:4px ; color:maroon ;">  &larr; add new room here</div>
          <div style="clear:both ; "></div>
        </div>
        <div id="dictionary-locations-rooms"></div>
      </div>
      <div style="clear:both ; "></div>
    </div>

    <!-- Statuses (dictionary) -->
    <div id="dictionary-statuses" class="application-workarea hidden">
      <div><button id="dictionary-statuses-reload" title="reload the dictionary from the database">Reload</button></div>
      <div style="float:left ;">
        <div style="margin-top:20px ;">
          <div style="float:left ; "><input type="text" size="12" name="status2add" title="fill in new status name, then press RETURN to save" /></div>
              <div style="float:left ; padding-top:4px ; color:maroon ;">  &larr; add new status here</div>
          <div style="clear:both ; "></div>
        </div>
        <div id="dictionary-statuses-statuses"></div>
      </div>
      <div style="float:left ; margin-left:20px ;">
        <div style="margin-top:20px ;">
          <div style="float:left ; "><input type="text" size="12" name="status22add" title="fill in new sub-status name, then press RETURN to save" /></div>
          <div style="float:left ; padding-top:4px ; color:maroon ;">  &larr; add new sub-status here</div>
          <div style="clear:both ; "></div>
        </div>
        <div id="dictionary-statuses-statuses2"></div>
      </div>
      <div style="clear:both ; "></div>
    </div>

    <!-- Access control -->
    <div id="admin-access" class="application-workarea hidden">
      <div style="float:left ;" ><button id="admin-access-reload" title="reload from the database">Reload</button></div>
      <div style="clear:both ; "></div>

      <div style="margin-top:20px ; margin-bottom:20px ; width:720px ;">
        <p>This section allows to assign user accounts to various roles defined in a context of the application.
        See a detailed description of each role in the corresponding subsection below.</p>
      </div>

      <div id="tabs" style="font-size:12px ;">
        <ul>
          <li><a href="#administrators">Administrators</a></li>
          <li><a href="#editors">Editors</a></li>
          <li><a href="#others">Other Users</a></li>
        </ul>

        <div id="administrators" >
          <div style="font-size:11px ; border:solid 1px #b0b0b0 ; padding:10px ; padding-left:20px ; padding-bottom:20px ;" >
            <div style="margin-bottom:10px ; width:720px ;">
              <p>Administrators posses highest level privileges in the application as they're allowed
              to perform any operation on the inventory and other users. The only restriction is that
              an administrator is not allowed to remove their own account from the list of administrators.</p>
            </div>
            <div style="float:left ; "><input type="text" size="8" name="administrator2add" title="fill in a UNIX account of a user, press RETURN to save" /></div>
            <div style="float:left ; padding-top: 4px ; color:maroon ; "> &larr; add new user here</div>
            <div style="clear:both ; "></div>
            <div id="admin-access-ADMINISTRATOR"></div>
          </div>
        </div>

        <div id="editors" >
          <div style="font-size:11px ; border:solid 1px #b0b0b0 ; padding:10px ; padding-left:20px ; padding-bottom:20px ;" >
            <div style="margin-bottom:10px ; width:720px ;">
              <p>Editors can add new equipment to the inventory, delete or edit existing records of the equipment
              and also manage certain aspects of the equipment life-cycle.</p>
            </div>
            <div style="float:left ; "><input type="text" size="8" name="editor2add" title="fill in a UNIX account of a user, press RETURN to save" /></div>
            <div style="float:left ; padding-top: 4px ; color:maroon ; "> &larr; add new user here</div>
            <div style="clear:both ; "></div>
            <div id="admin-access-EDITOR"></div>
          </div>
        </div>

        <div id="others" >
          <div style="font-size:11px ; border:solid 1px #b0b0b0 ; padding:10px ; padding-left:20px ; padding-bottom:20px ;" >
            <div style="margin-bottom:10px ; width:720px ;">
              <p>Other users may be allowed some limited access to manage certain aspects of the equipment life-cycle.</p>
            </div>
            <div style="float:left ; "><input type="text" size="8" name="other2add" title="fill in a UNIX account of a user, press RETURN to save" /></div>
            <div style="float:left ; padding-top: 4px ; color:maroon ; "> &larr; add new user here</div>
            <div style="clear:both ; "></div>
            <div id="admin-access-OTHER"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- E-mail notifications -->
    <div id="admin-notifications" class="application-workarea hidden">
      <div style="float:left ;" ><button id="admin-notifications-reload" title="reload from the database">Reload</button></div>
      <div style="clear:both ; "></div>

      <div style="margin-top:20px ; margin-bottom:20px ; width:720px ;">
        <p>In order to avoid an excessive e-mail traffic the notification system
        will send just one message for any modification made in a specific context. For the very same
        reason the default behavior of the system is to send a summary daily message with all changes
        made before a time specified below, unless this site administrators choose a different policy
        (such as instantaneous notification).</p>
       </div>

      <div id="tabs" style="font-size:12px ;">
        <ul>
          <li><a href="#myself">On my equipment</a></li>
          <li><a href="#administrators">Sent to administrators</a></li>
          <li><a href="#others">Sent to other users</a></li>
          <li><a href="#pending">Pending</a></li>
        </ul>

        <div id="myself" >
          <div style="font-size:11px ; border:solid 1px #b0b0b0 ; padding:10px ; padding-left:20px ; padding-bottom:20px ;" >
            <div style="margin-bottom:10px ; width:720px ;">
              <p>This section is aiming at editors who might be interested to track changes
              made to their equipment by other people involved into various stages
              of the workflow. Note that editors will not get notifications
              on changes made by themselves.</p>
              <p>Notification settings found in this section can only be managed by editors themselves
              or by administrators of the application.</p>
            </div>
            <div style="margin-bottom:20px ;">
              <select name="policy4EDITOR" disabled="disabled">
                <option value="DELAYED">daily notification (08:00am)</option>
                <option value="INSTANT">instant notification</option>
              </select>
            </div>
            <div id="admin-notifications-EDITOR"></div>
          </div>
        </div>

        <div id="administrators" >
          <div style="font-size:11px ; border:solid 1px #b0b0b0 ; padding:10px ; padding-left:20px ; padding-bottom:20px ;" >
            <div style="margin-bottom:10px ; width:720px ;">
              <p>This section is aiming at administrators of this software who might be interested to track major changes
              made to the equipment, user accounts or software configuration. Note that administrators will not get notifications
              on changes made by themselves.</p>
              <p>Notification settings found in this section can only be managed by any administrator of the software.</p>
            </div>
            <div style="margin-bottom:20px ;">
              <select name="policy4ADMINISTRATOR" disabled="disabled">
                <option value="DELAYED">daily notification (08:00am)</option>
                <option value="INSTANT">instant notification</option>
              </select>
            </div>
            <div id="admin-notifications-ADMINISTRATOR"></div>
          </div>
        </div>

        <div id="others" >
          <div style="font-size:11px ; border:solid 1px #b0b0b0 ; padding:10px ; padding-left:20px ; padding-bottom:20px ;" >
            <div style="margin-bottom:10px ; width:720px ;">
              <p>This section is aiming at users (not necessarily editors) who are involved
              into various stages of the equipment workflow.</p>
              <p>Only administrators of this application are allowed to modify notification settings found on this page.</p>
            </div>
            <div style="margin-bottom:20px ;">
              <select name="policy4OTHER" disabled="disabled">
                <option value="DELAYED">daily notification (08:00am)</option>
                <option value="INSTANT">instant notification</option>
              </select>
            </div>
            <div id="admin-notifications-OTHER"></div>
          </div>
        </div>

        <div id="pending" >
          <div style="font-size:11px ; border:solid 1px #b0b0b0 ; padding:10px ; padding-left:20px ; padding-bottom:20px ;" >
            <div style="margin-bottom:10px ; width:720px ;">
              <p>Pending/scheduled notifications (if any found below) can be submitted for instant delivery by pressing a group 'Submit' button or individually if needed.
              Notifications can also be deleted if needed. An additional dialog will be initiated to confirm group operations.</p>
              <p>Only administrators of this application are authorized for these operations.</p>
            </div>
            <div style="margin-bottom:20px ;"">
              <button name="submit_all" title="Submit all pending notifications to be instantly delivered to their recipient">submit</button>
              <button name="delete_all" title="Delete all pending notifications">delete</button>
            </div>
            <div id="admin-notifications-pending"></div>
          </div>
        </div>

      </div>
    </div>

    <!-- SLACid ranges -->
    <div id="admin-slacid" class="application-workarea hidden">
      <div style="float:left ;" ><button id="admin-slacid-reload" title="reload from the database">Reload</button></div>
      <div style="clear:both ; "></div>
      <div style="margin-top:20px ; margin-bottom:20px ; ">
        <p>PCDS is allocated a set of "official" SLACid numbers which are managed by
        this application. Each time a new equipment is being registered in the Inventory its proposed
        SLAC ID number will be validated to make sure it falls into one of the ranges.</p>
      </div>

      <div id="tabs" style="font-size:12px ;">
        <ul>
          <li><a href="#admin-slacid-ranges">Ranges</a></li>
        </ul>

        <div id="admin-slacid-ranges" >
          <div style="font-size:11px ; border:solid 1px #b0b0b0 ; padding:10px ; padding-left:20px ; padding-bottom:20px ;" >
            <div style="margin-bottom:10px ; ">
              <p>This page is meant to be used by administrators to manage known ranges.
                 Ranges can't overlap and they must be non-empty.</p>
            </div>
            <div id="ranges">
              <div style="margin-bottom:10px ;">
                <button name="edit">Edit</button>
                <button name="save">Save</button>
                <button name="cancel">Cancel</button>
              </div>
              <div id="admin-slacid-ranges-table"></div>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>

  <div id="popupdialogs" ></div>
  <div id="popupdialogs-varable-size" ></div>
  <div id="infodialogs" ></div>
  <div id="editdialogs" ></div>

</div>

</body>
</html>


<!--------------------- Document End Here -------------------------->

<?php

    $authdb->commit() ;
    $irep->commit() ;

} catch(AuthDBException   $e) { print $e->toHtml() ; }
  catch(LusiTimeException $e) { print $e->toHtml() ; }
  catch(IrepException     $e) { print $e->toHtml() ; }

?>
