define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk) {

    cssloader.load('../portal/css/Runtables_Calibrations.css') ;

    /**
     * The application for displaying and managing the run table for calibrations
     *
     * @returns {Runtables_Calibrations}
     */
    function Runtables_Calibrations (experiment, access_list) {

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

            this.container.html('<div id="runtables-calib"></div>') ;
            this.wa = this.container.find('div#runtables-calib') ;

            if (!this.access_list.runtables.read) {
                this.wa.html(this.access_list.no_page_access_html) ;
                return ;
            }

            var html =
'<div class="runtables-calib-ctrl"> ' +
  '<div class="runtables-calib-info" id="updated" style="float:right;" ></div> ' +
  '<div style="clear:both;" ></div> ' +
  '<table><tbody> ' +
    '<tr style="font-size:12px;" > ' +
      '<td valign="center"> ' +
        '<span style="font-weight:bold;">Select runs from</span> ' +
      '</td> ' +
      '<td valign="center"> ' +
        '<input type="text" name="from" size="2" title="' +
          'The first run of the interval. If no input is provided then ' +
          'the very first known run will be assumed." ' +
        '/> ' +
      '</td> ' +
      '<td valign="center" > ' +
        '<span style="font-weight:bold; margin-left:0px;" >through</span> ' +
      '</td>' +
      '<td valign="center" > ' +
        '<input name="through" type="text" size="2" title="' +
          'The last run of the interval. If no input is provided then the very ' +
          'last known run will be assumed" ' +
        '/ > ' +
      '</td>' +
      '<td valign="center" > ' +
        '<button class="control-button" style="margin-left:20px;" name="reset" title="reset the form" >RESET FORM</button> ' +
      '</td>' +
      '<td valign="center" > ' +
        '<button class="control-button update-button" name="update" title="check if there were any updates on this page" > ' +
          '<img src="../webfwk/img/Update.png" /> ' +
        '</button> ' +
      '</td> ' +
    '</tr> ' +
  '</tbody></table> ' +
'</div> ' +
'<div class="runtables-calib-wa" > ' +
'  <div class="runtables-calib-body" > ' +
'    <div id="table"></div> ' +
'  </div> ' +
'</div> ' ;
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

            var table_cont = this.wa.find('div#table') ;
            var hdr = [
                {   name: 'Run', type: Table.Types.Number_HTML } ,
                {   name: 'Dark', sorted: false ,
                    type: {
                        after_sort: function () {
                            table_cont.find('div.runtables-calib-dark').each(function () {
                                var elem = $(this) ;
                                var run = elem.attr('id') ;
                                var calib = that.runs[run] ;
                                elem.html('<span style="font-size:150%;">'+(calib.dark ? '&diams;' : '&nbsp;')+'</span>') ;
                            }) ;
                            if (that.access_list.runtables.edit_calibrations)
                                table_cont.find('input.runtables-calib-dark').each(function () {
                                    var elem = $(this) ;
                                    var run = elem.attr('id') ;
                                    var calib = that.runs[run] ;
                                    if (calib.dark) elem.attr('checked', 'checked') ;
                                    else            elem.removeAttr('checked') ;
                                }) ;
                        }
                    }
                } ,
                {   name: 'Flat', sorted: false ,
                    type: {
                        after_sort: function () {
                            table_cont.find('div.runtables-calib-flat').each(function () {
                                var elem = $(this) ;
                                var run = elem.attr('id') ;
                                var calib = that.runs[run] ;
                                elem.html('<span style="font-size:150%;">'+(calib.flat ? '&diams;' : '&nbsp;')+'</span>') ;
                            }) ;
                            if (that.access_list.runtables.edit_calibrations)
                                table_cont.find('input.runtables-calib-flat').each(function () {
                                    var elem = $(this) ;
                                    var run = elem.attr('id') ;
                                    var calib = that.runs[run] ;
                                    if (calib.flat) elem.attr('checked', 'checked') ;
                                    else            elem.removeAttr('checked') ;
                                }) ;
                        }
                    }
                } ,
                {   name: 'Geometry', sorted: false ,
                    type: {
                        after_sort: function () {
                            table_cont.find('div.runtables-calib-geom').each(function () {
                                var elem = $(this) ;
                                var run = elem.attr('id') ;
                                var calib = that.runs[run] ;
                                elem.html('<span style="font-size:150%;">'+(calib.geom ? '&diams;' : '&nbsp;')+'</span>') ;
                            }) ;
                            if (that.access_list.runtables.edit_calibrations)
                                table_cont.find('input.runtables-calib-geom').each(function () {
                                    var elem = $(this) ;
                                    var run = elem.attr('id') ;
                                    var calib = that.runs[run] ;
                                    if (calib.geom) elem.attr('checked', 'checked') ;
                                    else            elem.removeAttr('checked') ;
                                }) ;
                        }
                    }
                } ,
                {   name: 'Comment', sorted: false , hideable: true ,
                    type: {
                        after_sort: function () {
                            table_cont.find('div.runtables-calib-comment').each(function () {
                                var elem = $(this) ;
                                var run = elem.attr('id') ;
                                var calib = that.runs[run] ;
                                elem.html('<pre>'+calib.comment+'</pre>') ;
                            }) ;
                            if (that.access_list.runtables.edit_calibrations)
                                table_cont.find('textarea.runtables-calib-comment').each(function () {
                                    var elem = $(this) ;
                                    var run = elem.attr('id') ;
                                    var calib = that.runs[run] ;
                                    elem.text(calib.comment) ;
                                }) ;
                        }
                    }
                }
            ] ;
            if (this.access_list.runtables.edit_calibrations) hdr.push (
                {   name: 'ACTIONS', sorted: false , hideable: true ,
                    type: {
                        after_sort: function () {
                            table_cont.find('button.save_run').
                                button().
                                button('disable').
                                click(function () {

                                    var elem = $(this) ;
                                    var run = elem.attr('id') ;
                                    elem.button('disable') ;

                                    var dark = table_cont.find('input.runtables-calib-dark#'+run).attr('checked') ? 1 : 0 ;
                                    var flat = table_cont.find('input.runtables-calib-flat#'+run).attr('checked') ? 1 : 0 ;
                                    var geom = table_cont.find('input.runtables-calib-geom#'+run).attr('checked') ? 1 : 0 ;
                                    var comment = table_cont.find('textarea#'+run).val() ;

                                    that.save(run, dark, flat, geom, comment, function () {
                                        table_cont.find('button#'+run+'.save_run').button('disable') ;
                                        table_cont.find('.edit#'+run).parent().parent().parent().css('background-color', '') ;
                                    }) ;
                                }
                            ) ;
                            table_cont.find('.edit').change(function () {
                                var elem = $(this) ;
                                var run = elem.attr('id') ;
                                table_cont.find('button#'+run+'.save_run').button('enable') ;
                                $(this).parent().parent().parent().css('background-color', 'rgb(255, 220, 220)') ;
                            }) ;
                        }
                     }
                 }
            ) ;
            this.table = new Table (
                table_cont ,
                hdr ,
                null ,
                { default_sort_forward: false } ,
                Fwk.config_handler('runtables', 'calibrations')
            ) ;
            this.load() ;
        } ;

        this.display = function () {
            var title = 'show the run in the e-Log Search panel within the current Portal' ;
            var rows = [] ;
            for (var run in this.runs)
                rows.push(this.access_list.runtables.edit_calibrations ?
                    [   {number: run, html:   '<a href="javascript:global_elog_search_run_by_num('+run+',true);" title="'+title+'" class="link">'+run+'</a>'} ,
                        '<div class="runtables-calib-dark-cont"    ><input    class="runtables-calib-dark edit"    id="'+run+'" type="checkbox" /></div>' ,
                        '<div class="runtables-calib-flat-cont"    ><input    class="runtables-calib-flat edit"    id="'+run+'" type="checkbox" /></div>' ,
                        '<div class="runtables-calib-geom-cont"    ><input    class="runtables-calib-geom edit"    id="'+run+'" type="checkbox" /></div>' ,
                        '<div class="runtables-calib-comment-cont" ><textarea class="runtables-calib-comment edit" id="'+run+'" rows="2" cols="56" ></textarea></div>' ,
                        '<button class="save_run"   id="'+run+'" title="save modifications to the database" >Save</>'] :
                    [   {number: run, html:   '<a href="javascript:global_elog_search_run_by_num('+run+',true);" title="'+title+'" class="link">'+run+'</a>'} ,
                        '<div class="runtables-calib-dark-cont"    ><div      class="runtables-calib-dark"    id="'+run+'" ></div></div>' ,
                        '<div class="runtables-calib-flat-cont"    ><div      class="runtables-calib-flat"    id="'+run+'" ></div></div>' ,
                        '<div class="runtables-calib-geom-cont"    ><div      class="runtables-calib-geom"    id="'+run+'" ></div></div>' ,
                        '<div class="runtables-calib-comment-cont" ><div      class="runtables-calib-comment" id="'+run+'" ></div></div>' ]) ;

            this.table.load(rows) ;
            this.table.display() ;
        } ;

        this.load = function () {
            this.wa.find('.runtables-calib-info#updated').html('Loading...') ;
            Fwk.web_service_GET (
                '../portal/ws/runtable_calib_get.php' ,
                {   exper_id:    this.experiment.id ,
                    from_run:    parseInt(this.from_run.val()) ,
                    through_run: parseInt(this.through_run.val())
                } ,
                function (data) {
                    that.runs = data.runs ;
                    that.display() ;
                    that.wa.find('.runtables-calib-info#updated').html('Updated: <b>'+data.updated+'</b>') ;
                }
            ) ;
        } ;

        this.save = function (run, dark, flat, geom, comment, on_success) {
            this.wa.find('.runtables-calib-info#updated').html('Saving...') ;
            Fwk.web_service_POST (
                '../portal/ws/runtable_calib_save.php' , {
                    exper_id: this.experiment.id ,
                    run: run ,
                    dark: dark,
                    flat: flat,
                    geom: geom,
                    comment: comment
                } ,
                function (data) {
                    if (on_success) on_success() ;
                    else {
                        that.runs[run] = data.runs[run] ;
                        that.display() ;
                    }
                    that.wa.find('.runtables-calib-info#updated').html('[ Last update on: <b>'+data.updated+'</b> ]') ;
                }
            ) ;
        } ;
    }
    Class.define_class (Runtables_Calibrations, FwkApplication, {}, {}) ;

    return Runtables_Calibrations ;
}) ;
