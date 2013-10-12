<?php

require_once 'authdb/authdb.inc.php' ;

use AuthDb\AuthDb ;

AuthDb::instance()->begin() ;

$instruments = array('AMO','SXR','XPP','XCS','CXI','MEC') ;
$instr2editor = array() ;
foreach ($instruments as $instr_name) {
    $instr2editor[$instr_name] =
        AuthDb::instance()->hasRole (
            AuthDb::instance()->authName() ,
            null ,
            'ShiftMgr' ,
            "Manage_{$instr_name}"
        ) ;
}
?>

<!DOCTYPE html>
<html>

<head>

<title>Shift Manager</title>

<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">

<link type="text/css" href="/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css" rel="Stylesheet" />
<link type="text/css" href="/jquery/css/jquery-ui-timepicker-addon.css" rel="Stylesheet" />

<link type="text/css" href="../webfwk/css/Fwk.css" rel="Stylesheet" />
<link type="text/css" href="../webfwk/css/Stack.css" rel="Stylesheet" />
<link type="text/css" href="../webfwk/css/Table.css" rel="Stylesheet" />

<link type="text/css" href="../shiftmgr/css/shiftmgr.css" rel="Stylesheet" />

<style>

div.shift-reports,
div.shift-history-reports {
  padding: 20px;
}
</style>

<script type="text/javascript" src="/jquery/js/jquery-1.8.2.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui-1.9.1.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui-timepicker-addon.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.json.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.printElement.js"></script>

<script type="text/javascript" src="../webfwk/js/Class.js" ></script>
<script type="text/javascript" src="../webfwk/js/Widget.js" ></script>
<script type="text/javascript" src="../webfwk/js/StackOfRows.js" ></script>
<script type="text/javascript" src="../webfwk/js/Fwk.js"></script>
<script type="text/javascript" src="../webfwk/js/Table.js"></script>

<script type="text/javascript" src="../shiftmgr/js/Definitions.js"></script>
<script type="text/javascript" src="../shiftmgr/js/Reports.js"></script>
<script type="text/javascript" src="../shiftmgr/js/Reports4all.js"></script>
<script type="text/javascript" src="../shiftmgr/js/Access.js"></script>
<script type="text/javascript" src="../shiftmgr/js/Notifications.js"></script>
<script type="text/javascript" src="../shiftmgr/js/Rules.js"></script>
<script type="text/javascript" src="../shiftmgr/js/History.js"></script>

<script type="text/javascript">

$(function() {

    var instruments = <?php echo json_encode($instruments) ?> ;
    var instr2editor = <?php echo json_encode($instr2editor) ?> ;
    var menus = [] ;
    for (var i in instruments) {
        var instr_name = instruments[i] ;
        menus.push ({
            name: instr_name ,
            menu: [
                {   name: 'Reports' ,
                    application: new Reports(instr_name, instr2editor[instr_name]) ,
                    html_container: 'shift-reports-'+instr_name
                } ,
                {   name: 'History' ,
                    application: new History(instr_name) ,
                    html_container: 'shift-history-'+instr_name
                } ,
                {   name: 'E-mail Notifications' ,
                    application: new Notifications(instr_name) ,
                    html: 'View and manage push notifications: who will get an event and what kind of events (new shift created, data updated, etc.)'
                }
            ]
        }) ;
    }
    menus.push ({
        name: 'All Hutches',
        menu: [
            {   name: 'Reports' ,
                application: new Reports4all() ,
                html_container: 'shift-reports-all'
            } ,
            {   name: 'History' ,
                application: new History() ,
                html_container: 'shift-history-all'
            }
        ]
    }) ;
    menus.push ({
        name: 'Admin' ,
        menu: [
            {   name: 'Access Control' ,
                application: new Access() ,
                html: 'Access control management to assign priviles to individual users, groups, etc.'
            } ,
            {   name: 'Rules' ,
                application: new Rules() ,
                html: 'Various configuration parameters and rules, such: when shifts usuall begin, create shift placeholders automatically or not'
            }
        ]
    }) ;

    Fwk.build (

        // Title and subtitle (HTML allowed)

        'PCDS Shift Manager' ,  // title
        'Instrument Hutches' ,  // subtitle

        menus ,                 // menus and applications

        // The uick search provider

        function (text2search) { Fwk.report_info('Search', text2search) ; }
    ) ;
});

</script>

</head>

  <body>

      Loading...

<?php foreach ($instruments as $instr_name) { ?>

      <div id="shift-reports-<?php echo $instr_name ; ?>" style="display:none">

        <div class="shift-reports">

          <!-- Controls for selecting shifts for display and updating the list of
            -- the selected shifts. -->

          <div id="shifts-search-controls"  style="float:left;" >

            <div class="shifts-search-filters" >

              <div class="shifts-search-filter-group" >
                <div class="header" >Time range</div>
                <div class="cell-1" >
                  <select class="filter" name="range" style="padding:1px;">
                    <option value="week"  >Last 7 days</option>
                    <option value="month" >Last month</option>
                    <option value="range" >Specific range</option>
                  </select>
                </div>
                <div class="cell-2" >
                  <input class="filter" type="text" size=6 name="begin" disabled="disabled" title="specify the first day of the range (optional)" />
                  <input class="filter" type="text" size=6 name="end"  disabled="disabled" title="specify the last day of the range (optional)" />
                </div>
                <div class="terminator" ></div>
              </div>

              <div class="shifts-search-filter-group" >
                <div class="header" >Stopper out</div>
                <div class="cell-2">
                  <select class="filter" name="stopper" style="padding:1px;">
                    <option value=""  ></option>
                    <option value="0" >&gt; 0 %</option>
                    <option value="1" >&gt; 1 %</option>
                    <option value="2" >&gt; 2 %</option>
                    <option value="3" >&gt; 3 %</option>
                    <option value="4" >&gt; 4 %</option>
                    <option value="5" >&gt; 5 %</option>
                  </select>
                </div>
                <div class="header-1">Door open</div>
                <div class="cell-2" >
                  <select class="filter" name="door" style="padding:1px;">
                    <option value="" ></option>
                    <option value="100" >&lt; 100 %</option>
                    <option value="99"  >&lt; 99 %</option>
                    <option value="98"  >&lt; 98 %</option>
                    <option value="97"  >&lt; 97 %</option>
                    <option value="96"  >&lt; 96 %</option>
                    <option value="95"  >&lt; 95 %</option>
                  </select>
                </div>
                <div class="header-1" >LCLS beam</div>
                <div class="cell-2">
                  <select class="filter"t name="lcls" style="padding:1px;">
                    <option value=""  ></option>
                    <option value="0" >&gt; 0 %</option>
                    <option value="1" >&gt; 1 %</option>
                    <option value="2" >&gt; 2 %</option>
                    <option value="3" >&gt; 3 %</option>
                    <option value="4" >&gt; 4 %</option>
                    <option value="5" >&gt; 5 %</option>
                  </select>
                </div>
                <div class="header-1">Data taking</div>
                <div class="cell-2" >
                  <select class="filter" name="daq" style="padding:1px;">
                    <option value=""  ></option>
                    <option value="0" >&gt; 0 %</option>
                    <option value="1" >&gt; 1 %</option>
                    <option value="2" >&gt; 2 %</option>
                    <option value="3" >&gt; 3 %</option>
                    <option value="4" >&gt; 4 %</option>
                    <option value="5" >&gt; 5 %</option>
                  </select>
                </div>
                <div class="terminator" ></div>
              </div>

              <div class="shifts-search-filter-group" >
                <div class="header" >Shift types</div>
                <div class="cell-2">
                  <div class="cell-3" ><input class="filter type" type="checkbox" checked="checked" name="USER"     title="if enabled it will include shifts of this type" /></div><div class="cell-4">USER</div>
                  <div class="cell-3" ><input class="filter type" type="checkbox" checked="checked" name="MD"       title="if enabled it will include shifts of this type" /></div><div class="cell-4">MD</div>
                  <div class="cell-3" ><input class="filter type" type="checkbox" checked="checked" name="IN-HOUSE" title="if enabled it will include shifts of this type" /></div><div class="cell-4">IN-HOUSE</div>
                </div>
                <div class="terminator" ></div>
              </div>

            </div>
            <div class="shifts-search-buttons" >
              <button name="reset"  title="reset the search form to the default state">RESET</button>
            </div>
            <div class="shifts-search-filter-terminator" ></div>
          </div>

<?php if ($instr2editor[$instr_name]) { ?>
          <div id="new-shift-controls" style="float:left; margin-left:10px;">
            <button name="new-shift" title="open a dialog for creating a new shift" >CREATE NEW SHIFT</button>
            <div id="new-shift-con" class="new-shift-hdn" style="background-color:#f0f0f0; margin-top:5px; padding:1px 10px 5px 10px; border-radius:5px;" >
              <div style="max-width:460px;">
                <p>Note that shifts are usually created automatically based on rules defined
                in the Administrative section of this application. You may still want to create
                your own shift if that shift happens to be an exception from the rules.
                Possible cases would be: non-planned shift, very short shift, etc. In all
                other cases please see if there is a possibility to reuse an empty shift slot
                by checking "Display all shifts" checkbox on the left.</p>
              </div>
              <div style="float:left;">
                <table style="font-size:90%;"><tbody>
                  <tr>
                    <td class="shift-grid-hdr " valign="center" >Type:</td>
                    <td class="shift-grid-val " valign="center" >
                      <select name="type" >
                        <option value="USER"     >USER</option>
                        <option value="MD"       >MD</option>
                        <option value="IN-HOUSE" >IN-HOUSE</option>
                      </select></td>
                  </tr>
                  <tr>
                    <td class="shift-grid-hdr " valign="center" >Begin:</td>
                    <td class="shift-grid-val " valign="center" >
                      <input name="begin-day" type="text" size=8 title="specify the begin date of the shift" />
                      <input name="begin-h"   type="text" size=1 title="hour: 0..23" />
                      <input name="begin-m"   type="text" size=1 title="minute: 0..59" /></td>
                  </tr>
                  <tr>
                    <td class="shift-grid-hdr " valign="center" >End:</td>
                    <td class="shift-grid-val " valign="center" >
                      <input name="end-day" type="text" size=8 title="specify the end date of the shift" />
                      <input name="end-h"   type="text" size=1 title="hour: 0..23" />
                      <input name="end-m"   type="text" size=1 title="minute: 0..59" /></td>
                  </tr>
                </tbody></table>
              </div>
              <div style="float:left; margin-left:20px; margin-top:40px; padding-top:5px;">
                <button name="save"   title="submit modifications and open the editing dialog for the new shift">SAVE</button>
                <button name="cancel" title="discard modifications and close this dialog">CANCEL</button>
              </div>
              <div style="clear:both;"></div>
            </div>
          </div>
<?php } ?>
          <div style="clear:both;"></div>
          <div style="float:right;" id="shifts-search-info">Searching...</div>
          <div style="clear:both;"></div>

          <!-- The shifts display -->

          <div id="shifts-search-display"> </div>

        </div>
      </div>

      <div id="shift-history-<?php echo $instr_name ; ?>" style="display:none">

        <div class="shift-history-reports">

          <!-- Controls for selecting an interval for display and updating the list of
            -- the selected shifts. -->

          <div id="shifts-history-controls" style="float:left;" >
            <div>
              <table><tbody>
                <tr>
                  <td><b>Range:</b></td>
                  <td><select name="range" style="padding:1px;">
                        <option value="week"  >Last 7 days</option>
                        <option value="month" >Last month</option>
                        <option value="range" >Specific range</option>
                      </select></td>
                  <td><div style="width:20px;"></div>&nbsp;</td>
                  <td><input type="text" size=6 name="begin" disabled="disabled" title="specify the first day of the range (optional)" />
                      <b>&mdash;</b>
                      <input type="text" size=6 name="end"  disabled="disabled" title="specify the last day of the range (optional)" /></td>
                  <td><div style="width:20px;">&nbsp;</div></td>
                  <td><button name="reset"  title="reset the search form to the default state">Reset</button></td>
                </tr>
              </tbody></table>
            </div>
            <div style="margin-top:5px;" >
              <table><tbody>
                <tr>
                  <td><b>Display:</b></td>
                  <td class="annotated"
                      data="if enabled the table below will display shift creation events">
                    <input type="checkbox"
                           name="display-create-shift"
                           checked="checked" />CREATE SHIFT</td>
                </tr>
                <tr>
                  <td>&nbsp;</td>
                  <td class="annotated"
                      data="if enabled the table below will display shift modifications">
                    <input type="checkbox"
                           name="display-modify-shift"
                           checked="checked" />MODIFY SHIFT</td>
                  <td><div style="width:20px;"></div>&nbsp;</td>
                  <td class="annotated"
                      data="if enabled the table below will display area modifications">
                    <input type="checkbox"
                           name="display-modify-area"
                           checked="checked" />MODIFY AREA</td>
                  <td><div style="width:20px;"></div>&nbsp;</td>
                  <td class="annotated"
                      data="if enabled the table below will display timer allocation modifications">
                    <input type="checkbox"
                           name="display-modify-time"
                           checked="checked" />MODIFY TIME ALLOCATION</td>
                </tr>
              </tbody></table>
            </div>
          </div>
          <div style="clear:both;"></div>
          <div style="float:right;" id="shifts-history-info">Searching...</div>
          <div style="clear:both;"></div>

          <!-- The shifts display -->

          <div id="shifts-history-display"></div>

        </div>
      </div>
<?php } ?>


      <!-- Across all instruments -->

      <div id="shift-reports-all" style="display:none">

        <div class="shift-reports">

          <!-- Controls for selecting shifts for display and updating the list of
            -- the selected shifts. -->

          <div id="shifts-search-controls" >

            <div class="shifts-search-filters" >

              <div class="shifts-search-filter-group" >
                <div class="header" >Time range</div>
                <div class="cell-1" >
                  <select class="filter" name="range" style="padding:1px;">
                    <option value="week"  >Last 7 days</option>
                    <option value="month" >Last month</option>
                    <option value="range" >Specific range</option>
                  </select>
                </div>
                <div class="cell-2" >
                  <input class="filter" type="text" size=6 name="begin" disabled="disabled" title="specify the first day of the range (optional)" />
                  <input class="filter" type="text" size=6 name="end"  disabled="disabled" title="specify the last day of the range (optional)" />
                </div>
                <div class="terminator" ></div>
              </div>

              <div class="shifts-search-filter-group" >
                <div class="header" >Stopper out</div>
                <div class="cell-2">
                  <select class="filter" name="stopper" style="padding:1px;">
                    <option value=""  ></option>
                    <option value="0" >&gt; 0 %</option>
                    <option value="1" >&gt; 1 %</option>
                    <option value="2" >&gt; 2 %</option>
                    <option value="3" >&gt; 3 %</option>
                    <option value="4" >&gt; 4 %</option>
                    <option value="5" >&gt; 5 %</option>
                  </select>
                </div>
                <div class="header-1">Door open</div>
                <div class="cell-2" >
                  <select class="filter" name="door" style="padding:1px;">
                    <option value="" ></option>
                    <option value="100" >&lt; 100 %</option>
                    <option value="99"  >&lt; 99 %</option>
                    <option value="98"  >&lt; 98 %</option>
                    <option value="97"  >&lt; 97 %</option>
                    <option value="96"  >&lt; 96 %</option>
                    <option value="95"  >&lt; 95 %</option>
                  </select>
                </div>
                <div class="header-1" >LCLS beam</div>
                <div class="cell-2">
                  <select class="filter"t name="lcls" style="padding:1px;">
                    <option value=""  ></option>
                    <option value="0" >&gt; 0 %</option>
                    <option value="1" >&gt; 1 %</option>
                    <option value="2" >&gt; 2 %</option>
                    <option value="3" >&gt; 3 %</option>
                    <option value="4" >&gt; 4 %</option>
                    <option value="5" >&gt; 5 %</option>
                  </select>
                </div>
                <div class="header-1">Data taking</div>
                <div class="cell-2" >
                  <select class="filter" name="daq" style="padding:1px;">
                    <option value=""  ></option>
                    <option value="0" >&gt; 0 %</option>
                    <option value="1" >&gt; 1 %</option>
                    <option value="2" >&gt; 2 %</option>
                    <option value="3" >&gt; 3 %</option>
                    <option value="4" >&gt; 4 %</option>
                    <option value="5" >&gt; 5 %</option>
                  </select>
                </div>
                <div class="terminator" ></div>
              </div>

              <div class="shifts-search-filter-group" >
                <div class="header" >Instruments</div>
                <div class="cell-2">
                  <div class="cell-3" ><input class="filter instrument" type="checkbox" checked="checked" name="AMO" title="if enabled it will include shifts from the instrument " /></div><div class="cell-4">AMO</div>
                  <div class="cell-3" ><input class="filter instrument" type="checkbox" checked="checked" name="SXR" title="if enabled it will include shifts from the instrument " /></div><div class="cell-4">SXR</div>
                  <div class="cell-3" ><input class="filter instrument" type="checkbox" checked="checked" name="XPP" title="if enabled it will include shifts from the instrument " /></div><div class="cell-4">XPP</div>
                  <div class="cell-3" ><input class="filter instrument" type="checkbox" checked="checked" name="XCS" title="if enabled it will include shifts from the instrument " /></div><div class="cell-4">XCS</div>
                  <div class="cell-3" ><input class="filter instrument" type="checkbox" checked="checked" name="CXI" title="if enabled it will include shifts from the instrument " /></div><div class="cell-4">CXI</div>
                  <div class="cell-3" ><input class="filter instrument" type="checkbox" checked="checked" name="MEC" title="if enabled it will include shifts from the instrument " /></div><div class="cell-4">MEC</div>
                  <div class="terminator" ></div>
                </div>
                <div class="terminator" ></div>
              </div>

              <div class="shifts-search-filter-group" >
                <div class="header" >Shift types</div>
                <div class="cell-2">
                  <div class="cell-3" ><input class="filter type" type="checkbox" checked="checked" name="USER"     title="if enabled it will include shifts of this type" /></div><div class="cell-4">USER</div>
                  <div class="cell-3" ><input class="filter type" type="checkbox" checked="checked" name="MD"       title="if enabled it will include shifts of this type" /></div><div class="cell-4">MD</div>
                  <div class="cell-3" ><input class="filter type" type="checkbox" checked="checked" name="IN-HOUSE" title="if enabled it will include shifts of this type" /></div><div class="cell-4">IN-HOUSE</div>
                </div>
                <div class="terminator" ></div>
              </div>

            </div>
            <div class="shifts-search-buttons" >
              <button name="reset"  title="reset the search form to the default state">RESET</button>
              <button name="search" title="search again">SEARCH AGAIN</button>
            </div>
            <div class="shifts-search-filter-terminator" ></div>
          </div>

          <div style="float:right;" id="shifts-search-info">Searching...</div>
          <div style="clear:both;"></div>

          <!-- The shifts display -->

          <div id="shifts-search-display"></div>

        </div>
      </div>

      <div id="shift-history-all" style="display:none">

        <div class="shift-history-reports">

          <!-- Controls for selecting an interval for display and updating the list of
            -- the selected shifts. -->

          <div id="shifts-history-controls" style="float:left;" >
            <div>
              <table><tbody>
                <tr>
                  <td><b>Range:</b></td>
                  <td><select name="range" style="padding:1px;">
                        <option value="week"  >Last 7 days</option>
                        <option value="month" >Last month</option>
                        <option value="range" >Specific range</option>
                      </select></td>
                  <td><div style="width:20px;"></div>&nbsp;</td>
                  <td><input type="text" size=6 name="begin" disabled="disabled" title="specify the first day of the range (optional)" />
                      <b>&mdash;</b>
                      <input type="text" size=6 name="end"  disabled="disabled" title="specify the last day of the range (optional)" /></td>
                  <td><div style="width:20px;">&nbsp;</div></td>
                  <td><button name="reset"  title="reset the search form to the default state">Reset</button></td>
                </tr>
              </tbody></table>
            </div>
            <div style="margin-top:5px;" >
              <table><tbody>
                <tr>
                  <td><b>Display:</b></td>
                  <td class="annotated"
                      data="if enabled the table below will display shift creation events">
                    <input type="checkbox"
                           name="display-create-shift"
                           checked="checked" />CREATE SHIFT</td>
                </tr>
                <tr>
                  <td>&nbsp;</td>
                  <td class="annotated"
                      data="if enabled the table below will display shift modifications">
                    <input type="checkbox"
                           name="display-modify-shift"
                           checked="checked" />MODIFY SHIFT</td>
                  <td><div style="width:20px;"></div>&nbsp;</td>
                  <td class="annotated"
                      data="if enabled the table below will display area modifications">
                    <input type="checkbox"
                           name="display-modify-area"
                           checked="checked" />MODIFY AREA</td>
                  <td><div style="width:20px;"></div>&nbsp;</td>
                  <td class="annotated"
                      data="if enabled the table below will display timer allocation modifications">
                    <input type="checkbox"
                           name="display-modify-time"
                           checked="checked" />MODIFY TIME ALLOCATION</td>
                </tr>
              </tbody></table>
            </div>
          </div>
          <div style="clear:both;"></div>
          <div style="float:right;" id="shifts-history-info">Searching...</div>
          <div style="clear:both;"></div>

          <!-- The shifts display -->

          <div id="shifts-history-display"></div>

        </div>
      </div>

  </body>

</html>

