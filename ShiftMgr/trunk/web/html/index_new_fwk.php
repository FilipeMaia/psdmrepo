<?php
$instruments = array('AMO','SXR','XPP','XCS','CXI','MEC') ;
?>

<!DOCTYPE html>
<html>

<head>

<title>Shift Manager</title>

<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1">

<link type="text/css" href="/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css" rel="Stylesheet" />
<link type="text/css" href="/jquery/css/jquery-ui-timepicker-addon.css" rel="Stylesheet" />

<link type="text/css" href="../portal/css/Fwk.css" rel="Stylesheet" />
<link type="text/css" href="../portal/css/Table.css" rel="Stylesheet" />

<link type="text/css" href="../shiftmgr/css/shiftmgr.css" rel="Stylesheet" />

<style>

div.shift_reports {
  padding: 20px;
}
</style>

<script type="text/javascript" src="/jquery/js/jquery-1.8.2.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui-1.9.1.custom.min.js"></script>
<script type="text/javascript" src="/jquery/js/jquery-ui-timepicker-addon.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.form.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.json.js"></script>
<script type="text/javascript" src="/jquery/js/jquery.printElement.js"></script>

<script type="text/javascript" src="../portal/js/Fwk.js"></script>
<script type="text/javascript" src="../portal/js/Table.js"></script>

<script type="text/javascript" src="../shiftmgr/js/Reports.js"></script>
<script type="text/javascript" src="../shiftmgr/js/Analytics.js"></script>
<script type="text/javascript" src="../shiftmgr/js/Access.js"></script>
<script type="text/javascript" src="../shiftmgr/js/Notifications.js"></script>
<script type="text/javascript" src="../shiftmgr/js/Rules.js"></script>

<script type="text/javascript">

$(function() {

    var instruments = <?php echo json_encode($instruments) ?> ;
    var menus = [] ;
    for (var i in instruments) {
        var instr_name = instruments[i] ;
        menus.push ({
            name: instr_name ,
            menu: [
                {   name: 'Reports' ,
                    dispatcher: new Reports(instr_name) ,
                    html_container: 'shift_reports_'+instr_name
                } ,
                {   name: 'Analytics' ,
                    dispatcher: new Analytics(instr_name) ,
                    html_container: 'shift_analytics_'+instr_name
                }
            ]
        }) ;
    }
    menus.push ({
        name: 'Admin' ,
        menu: [
            {   name: 'Access' ,
                dispatcher: new Access() ,
                html: 'Access control management to assign priviles to individual users, groups, etc.'
            } ,
            {   name: 'Notifications' ,
                dispatcher: new Notifications() ,
                html: 'View and manage push notifications: who will get an event and what kind of events (new shift created, data updated, etc.)'
            } ,
            {   name: 'Rules' ,
                dispatcher: new Rules() ,
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

      <div id="shift_reports_<?php echo $instr_name ; ?>" style="display:none">

        <div class="shift_reports">

          <!-- Controls for selecting shifts for display and updating the list of
            -- the selected shifts. -->

          <div id="shifts-search-controls">
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

          <div style="float:right;" id="shifts-search-info">Searching...</div>
          <div style="clear:both;"></div>

          <!-- The shifts display -->

          <div id="shifts-search-display">

            <!-- Table header -->

            <div id="shifts-search-header">
              <div style="float:left; margin-left: 0px;                                 width: 20px;" ><div  class="shift-toggler"  >&nbsp;  </div></div>
              <div style="float:left; margin-left:10px; text-align: right;              width: 80px;" ><span class="shift-table-hdr">Shift   </span></div>
              <div style="float:left; margin-left:20px;                                 width: 40px;" ><span class="shift-table-hdr">begin   </span></div>
              <div style="float:left; margin-left:10px;                                 width: 40px;" ><span class="shift-table-hdr">end     </span></div>
              <div style="float:left; margin-left:20px; border-right:1px solid #000000; width: 50px;" ><span class="shift-table-hdr">&Delta;t</span></div>
              <div style="float:left; margin-left:10px;                                 width: 70px;" ><span class="shift-table-hdr">Stopper </span></div>
              <div style="float:left; margin-left:10px; border-right:1px solid #000000; width: 50px;" ><span class="shift-table-hdr">Door    </span></div>
              <div style="float:left; margin-left:15px;                                 width: 40px;" ><span class="shift-table-hdr">FEL     </span></div>
              <div style="float:left; margin-left: 5px;                                 width: 40px;" ><span class="shift-table-hdr">BMLN    </span></div>
              <div style="float:left; margin-left: 5px;                                 width: 40px;" ><span class="shift-table-hdr">CTRL    </span></div>
              <div style="float:left; margin-left: 5px;                                 width: 40px;" ><span class="shift-table-hdr">DAQ     </span></div>
              <div style="float:left; margin-left: 5px;                                 width: 40px;" ><span class="shift-table-hdr">LASR    </span></div>
              <div style="float:left; margin-left: 5px;                                 width: 40px;" ><span class="shift-table-hdr">HALL    </span></div>
              <div style="float:left; margin-left: 5px;                                 width: 50px;" ><span class="shift-table-hdr">OTHR    </span></div>
              <div style="float:left; margin-left: 5px; padding-left: 20px; border-left:1px solid #000000;  width: 50px;" ><span class="shift-table-hdr">Editor  </span></div>
              <div style="float:left; margin-left:10px;                                 width:140px;" ><span class="shift-table-hdr">Modified</span></div>
              <div style="clear:both;"></div>
            </div>

            <!-- Table body is loaded dynamically by the application -->

            <div id="shifts-search-list">
              <div style="color:maroon; margin-top:10px;">
                Use the search form to find shifts...
              </div>
            </div>

          </div>
        </div>
      </div>

      <div id="shift_analytics_<?php echo $instr_name ; ?>" style="display:none">
        Shift analytics for <?php echo $instr_name ; ?>
      </div>

<?php } ?>

  </body>

</html>

