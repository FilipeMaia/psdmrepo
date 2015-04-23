define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk) {

    cssloader.load('../portal/css/Runtables_Detectors.css') ;

    /**
     * The application for displaying the run table with DAQ detectors
     *
     * @returns {Runtables_Detectors}
     */
    function Runtables_Detectors (experiment, access_list) {

        var that = this ;

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        FwkApplication.call(this) ;

        // ------------------------------------------------
        // Override event handler defined in the base class
        // ------------------------------------------------

        this.on_activate = function() {
            this.on_update() ;
        } ;

        this.on_deactivate = function() {
            this.init() ;
        } ;

        this.on_update = function () {
            if (this.active) {
                this.init() ;
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this.experiment  = experiment ;
        this.access_list = access_list ;

        // --------------------
        // Own data and methods
        // --------------------

        this.is_initialized = false ;

        this.wa = null ;

        this.runs = null ;

        this.update_button = null ;
        this.reset_button = null ;
        this.from_run = null ;
        this.through_run = null ;

        this.table = null ;

        this.init = function () {

            if (this.is_initialized) return ;
            this.is_initialized = true ;

            this.container.html('<div id="runtables-detectors"></div>') ;
            this.wa = this.container.find('div#runtables-detectors') ;

            if (!this.access_list.runtables.read) {
                this.wa.html(this.access_list.no_page_access_html) ;
                return ;
            }

            var html =
'<div class="runtables-detectors-ctrl">' +
'  <div class="runtables-detectors-info" id="updated" style="float:right;"></div>' +
'  <div style="clear:both;"></div> ' +
'  <table><tbody>' +
'    <tr style="font-size:12px;">' +
'      <td valign="center">' +
'        <span style="font-weight:bold;">Select runs from</span></td>' +
'      <td valign="center">' +
'        <input type="text" name="from" size="2" title="The first run of the interval. If no input is provided then the very first known run will be assumed." /></td>' +
'      <td valign="center">' +
'        <span style="font-weight:bold; margin-left:0px;">through</span></td>' +
'      <td valign="center">' +
'        <input name="through" type="text" size="2" title="The last run of the interval. If no input is provided then the very last known run will be assumed"/ ></td>' +
'      <td valign="center">' +
'        <button class="control-button" style="margin-left:20px;" name="reset" title="reset the form">RESET FORM</button></td>' +
'      <td valign="center">' +
'        <button class="control-button update-button" name="update" title="check if there were any updates on this page"><img src="../webfwk/img/Update.png" /></button></td>' +
'    </tr>' +
'  </tbody></table>' +
'</div>' +
'<div class="runtables-detectors-wa">' +
'  <div class="runtables-detectors-body">' +
'    <div id="table-controls" style="margin-bottom:10px;">' +
'      <table><tbody>' +
'        <tr style="font-size:12px;">' +
'          <td valign="center">' +
'            <button class="control-button" name="advanced" title="open a dialog to select which columns to show/hide">SELECT DETECTORS</td>' +
'          <td valign="center">' +
'            <button class="control-button" name="show_all" title="show all columns">SHOW ALL</button></td>' +
'          <td valign="center">' +
'            <button class="control-button" name="hide_all" title="hide all columns">HIDE ALL</button></td>' +
'        </tr>' +
'      </tbody></table>' +
'    </div>' +
'    <div id="table"></div>' +
'  </div>' +
'</div>' ;
            this.wa.html(html) ;

            this.update_button = this.wa.find('button[name="update"]').button() ;
            this.reset_button  = this.wa.find('button[name="reset"]')  .button() ;
            this.from_run      = this.wa.find('input[name="from"]') ;
            this.through_run   = this.wa.find('input[name="through"]') ;

            this.update_button.click (function () { that.load() ;  }) ;
            this.from_run     .change(function () { that.load() ;  }) ;
            this.through_run  .change(function () { that.load() ;  }) ;

            this.reset_button.click (function () {
                that.from_run.val('') ;
                that.through_run.val('') ;
                that.load() ;
            }) ;

            this.show_all_button = this.wa.find('button[name="show_all"]').button().button('disable') ;
            this.hide_all_button = this.wa.find('button[name="hide_all"]').button().button('disable') ;
            this.advanced_button = this.wa.find('button[name="advanced"]').button().button('disable') ;

            this.show_all_button.click(function() { that.table.display('show_all') ; }) ;
            this.hide_all_button.click(function() { that.table.display('hide_all') ; }) ;
            this.advanced_button.click(function() { that.advanced() ; }) ;

            this.load() ;
        } ;

        this.advanced = function () {

            var detectors_selector = $('#fwk-largedialogs') ;

            var html =
'<div style="overflow:auto;">' +
'  <table><tbody>' ;
            var detectors_per_row = 5 ;
            var num_detectors = 0 ;
            var header_info = this.table.header_info() ;
            for (var i in header_info) {
                var col = header_info[i] ;
                if (col.hideable) {
                    if (!num_detectors) {
                        html +=
'    <tr>' ;
                    } else if (!(num_detectors % detectors_per_row)) {
                        html +=
'    </tr>' +
'    <tr>' ;
                    }
                    html +=
'      <td class="table_cell table_cell_borderless">' +
'        <div style="float:left;"><input type="checkbox" class="detector" name="'+col.number+'" '+(col.hidden?'':'checked="checked"')+' /></div>' +
'        <div style="float:left; margin-left:5px; font-weight:bold;">'+col.name+'</div>' +
'        <div style="clear:both;"></div>' +
'      </td>' ;
                    num_detectors++ ;
                }
            }
            if (num_detectors % detectors_per_row) {
                html +=
'    </tr>' ;
            }
            html +=
'  </tbody></table>' ;
'</div>';
            detectors_selector.html (html) ;
            detectors_selector.dialog ({
                minHeight: 240 ,
                width: 720 ,
                resizable: true ,
                modal: true ,
                buttons: {
                    "Close": function() {
                        $( this ).dialog('close') ;
                    }
                } ,
                title: 'Switch on/off detectors'
            });
            detectors_selector.find('input.detector').change(function() {
                var col_number = this.name ;
                that.table.display(this.checked ? 'show' : 'hide', parseInt(col_number)) ;
            }) ;
        } ;
        this.display = function () {
            var table_cont = this.wa.find('div#table') ;
            var hdr = [
                {name: 'RUN', type: Table.Types.Number_HTML}
            ] ;
            for (var name in that.detectors) {
                var det_dev = name.split('|') ;
                var det = det_dev[0] ;
                var dev = det_dev[1] ;
                hdr.push({
                    name: '<div><div>'+det+'</div><div>'+dev+'</div></div>' ,
                    hideable: true ,
                    align: 'center' ,
                    style: ' white-space: nowrap;'
                }) ;
            }
            this.table = new Table (
                table_cont ,
                hdr ,
                null ,
                { default_sort_forward: false } ,
                Fwk.config_handler('runtables', 'detectors')
            ) ;
            var title = 'show the run in the e-Log Search panel within the current Portal' ;
            var rows = [] ;
            for (var run in this.runs) {
                var row = [] ;
                row.push(
                    {   number: run ,
                        html:   '<a href="javascript:global_elog_search_run_by_num('+run+',true);" title="'+title+'" class="link">'+run+'</a>'
                    }
                ) ;
                var run_detectors = this.runs[run] ;
                for (var name in that.detectors) {
                    row.push('<span style="font-size:150%;">'+(run_detectors[name] ? '&diams;' : '&nbsp;')+'</span>')  ;
                }
                rows.push(row) ;
            }
            this.table.load(rows) ;
            this.table.display() ;
        } ;

        this.load = function () {
            this.wa.find('.runtables-detectors-info#updated').html('Loading...') ;
            that.show_all_button.button('disable') ;
            that.hide_all_button.button('disable') ;
            that.advanced_button.button('disable') ;
            Fwk.web_service_GET (
                '../portal/ws/runtable_detectors_get.php' ,
                {   exper_id:    this.experiment.id ,
                    from_run:    parseInt(this.from_run.val()) ,
                    through_run: parseInt(this.through_run.val())
                } ,
                function (data) {
                    that.detectors      = data.detectors ;
                    that.runs = data.runs ;
                    that.display() ;
                    that.wa.find('.runtables-detectors-info#updated').html('Updated: <b>'+data.updated+'</b>') ;
                    that.show_all_button.button('enable') ;
                    that.hide_all_button.button('enable') ;
                    that.advanced_button.button('enable') ;
                }
            ) ;
        } ;
    }
    Class.define_class (Runtables_Detectors, FwkApplication, {}, {}) ;

    return Runtables_Detectors ;
}) ;
