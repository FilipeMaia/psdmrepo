/*
 * ======================================================
 *  Application: Run Tables
 *  DOM namespace of classes and identifiers: runtables-
 *  JavaScript global names begin with: runtables-
 * ======================================================
 */

function runtables_create () {

    var that = this ;

    /* ---------------------------------------
     *  This application specific environment
     * ---------------------------------------
     *
     * NOTE: These variables should be initialized externally.
     */
    this.exp_id  = null ;

    /* The context for v-menu items
     */
    var context2_default = {
        'calib' : ''
    } ;
    this.name = 'runtables' ;
    this.full_name = 'Run Tables' ;
    this.context1 = 'calib' ;
    this.context2 = '' ;
    this.select_default = function () { this.select(this.context1, this.context2) ; } ;
    this.select = function (ctx1, ctx2) {
        this.init() ;
        this.context1 = ctx1 ;
        this.context2 = ctx2 == null ? context2_default[ctx1] : ctx2 ;
        if (this.context1 === 'calib') {
            this.calib_update() ;
        }
    } ;

    /* ---------------------------------------
     *  Menu item: Calibration control table
     * ---------------------------------------
     */
    this.calib_runs = null ;

    this.calib_refresh_button = null ;
    this.calib_reset_button = null ;
    this.calib_from_run = null ;
    this.calib_through_run = null ;

    this.calib_table = null ;

    this.calib_init = function () {

        var that = this ;

        this.calib_refresh_button = $('#runtables-calib').find('button[name="refresh"]').button() ;
        this.calib_reset_button   = $('#runtables-calib').find('button[name="reset"]')  .button() ;
        this.calib_from_run       = $('#runtables-calib').find('input[name="from"]') ;
        this.calib_through_run    = $('#runtables-calib').find('input[name="through"]') ;

        this.calib_refresh_button.click (function () { that.calib_update() ;  }) ;
        this.calib_from_run      .change(function () { that.calib_update() ;  }) ;
        this.calib_through_run   .change(function () { that.calib_update() ;  }) ;

        this.calib_reset_button.click (function () {
            that.calib_from_run.val('') ;
            that.calib_through_run.val('') ;
            that.calib_update() ;
        }) ;

        var table_cont = $('#runtables-calib').find('div#table') ;
        var hdr = [
            {   name: 'Run', type: Table.Types.Number_HTML } ,
            {   name: 'Dark', sorted: false ,
                type: {
                    after_sort: function () {
                        table_cont.find('div.runtables-calib-dark').each(function () {
                            var elem = $(this) ;
                            var run = elem.attr('id') ;
                            var calib = that.calib_runs[run] ;
                            elem.html(calib.dark ? '<span style="color:red;">Yes</span>' : '') ;
                        }) ;
                        table_cont.find('input.runtables-calib-dark').each(function () {
                            var elem = $(this) ;
                            var run = elem.attr('id') ;
                            var calib = that.calib_runs[run] ;
                            if (calib.dark) elem.attr('checked', 'checked') ;
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
                            var calib = that.calib_runs[run] ;
                            elem.html('<pre>'+calib.comment+'</pre>') ;
                        }) ;
                        table_cont.find('textarea.runtables-calib-comment').each(function () {
                            var elem = $(this) ;
                            var run = elem.attr('id') ;
                            var calib = that.calib_runs[run] ;
                            elem.text(calib.comment) ;
                        }) ;
                    }
                }
            } ,
            {   name: 'ACTIONS', sorted: false , hideable: true ,
                type: {
                    after_sort: function () {
                        table_cont.find('button.edit_run').
                            button().
                            click(function () {

                                var elem = $(this) ;
                                var run = elem.attr('id') ;
                                elem.button('disable') ;

                                table_cont.find('button.cancel_run#'+run).button('enable') ;
                                table_cont.find('button.save_run#'+run).button('enable') ;


                                table_cont.find('.view#'+run).addClass('runtables-calib-hdn').removeClass('runtables-calib-vis') ;
                                table_cont.find('.edit#'+run).addClass('runtables-calib-vis').removeClass('runtables-calib-hdn') ;
                            }
                        ) ;
                        table_cont.find('button.cancel_run').
                            button().
                            button('disable').
                            click(function () {

                                var elem = $(this) ;
                                var run = elem.attr('id') ;
                                elem.button('disable') ;

                                table_cont.find('button.edit_run#'+run).button('enable') ;
                                table_cont.find('button.save_run#'+run).button('disable') ;

                                table_cont.find('.view#'+run).addClass('runtables-calib-vis').removeClass('runtables-calib-hdn') ;
                                table_cont.find('.edit#'+run).addClass('runtables-calib-hdn').removeClass('runtables-calib-vis') ;
                            }
                        ) ;
                        table_cont.find('button.save_run').
                            button().
                            button('disable').
                            click(function () {

                                var elem = $(this) ;
                                var run = elem.attr('id') ;
                                elem.button('disable') ;

                                table_cont.find('button.edit_run#'+run).button('enable') ;
                                table_cont.find('button.cancel_run#'+run).button('disable') ;

                                table_cont.find('.view#'+run).addClass('runtables-calib-vis').removeClass('runtables-calib-hdn') ;
                                table_cont.find('.edit#'+run).addClass('runtables-calib-hdn').removeClass('runtables-calib-vis') ;

                                var dark = table_cont.find('input#'+run).attr('checked') ? 1 : 0 ;
                                var comment = table_cont.find('textarea#'+run).val() ;

                                that.calib_save(run, dark, comment) ;
                            }
                        ) ;
                    }
                 }
             }
        ] ;
        this.calib_table = new Table (
            table_cont ,
            hdr ,
            null ,
            { default_sort_forward: false } ,
            config.handler('runtables', 'calibrations')
        ) ;
        this.calib_update() ;
    } ;

    this.calib_display = function () {
        var title = 'show the run in the e-Log Search panel within the current Portal' ;
        var rows = [] ;
        for (var run in this.calib_runs) {
            var run_url = '<a href="javascript:global_elog_search_run_by_num('+run+',true);" title="'+title+'" class="lb_link"><img src="../portal/img/link2run_32x32.png" /></a>' ;
            rows.push([
                {   number: run ,
                    html:   '<div style="float:left;">'+run_url+'</div><div style="float:left; padding-left:8px; padding-top:8px;">'+run+'</div><div style="clear:both;"></div>'
                } ,
                '<div class="runtables-calib-dark-cont" >' +
                '  <div      class="runtables-calib-dark view runtables-calib-vis" id="'+run+'" ></div>' +
                '  <input    class="runtables-calib-dark edit runtables-calib-hdn" id="'+run+'" type="checkbox" />' +
                '</div>',
                '<div class="runtables-calib-comment-cont" >' +
                '  <div      class="runtables-calib-comment view runtables-calib-vis" id="'+run+'" ></div>' +
                '  <textarea class="runtables-calib-comment edit runtables-calib-hdn" id="'+run+'" rows="2" cols="56" ></textarea>' +
                '</div>',
                '<button class="edit_run"   id="'+run+'" title="edit calibration properties of this run"  >Edit</>' +
                '<button class="cancel_run" id="'+run+'" title="discard modifications (if any were made)" >Cancel</>' +
                '<button class="save_run"   id="'+run+'" title="save modifications to the database"       >Save</>'
            ]) ;
        }
        this.calib_table.load(rows) ;
        this.calib_table.display() ;
    } ;

    this.calib_update = function () {
        $('#runtables-calib').find('.runtables-info#updated').html('Loading...') ;
        web_service_GET (
            '../portal/ws/runtable_calib_get.php' ,
            {   exper_id:    this.exp_id ,
                from_run:    parseInt(this.calib_from_run.val()) ,
                through_run: parseInt(this.calib_through_run.val())
            } ,
            function (data) {
                that.calib_runs = data.runs ;
                that.calib_display() ;
                $('#runtables-calib').find('.runtables-info#updated').html('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }
        ) ;
    } ;

    this.calib_save = function (run, dark, comment) {
        $('#runtables-calib').find('.runtables-info#updated').html('Saving...') ;
        web_service_POST (
            '../portal/ws/runtable_calib_save.php' , {
                exper_id: this.exp_id ,
                run: run ,
                dark: dark,
                comment: comment
            } ,
            function (data) {
                that.calib_runs[run] = data.runs[run] ;
                that.calib_display() ;
                $('#runtables-calib').find('.runtables-info#updated').html('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }
        ) ;    } ;

    /* ----------------------------------
     *  Application initialization point
     * ----------------------------------
     *
     * RETURN: true if the real initialization took place. Return false
     *         otherwise.
     */
    this.is_initialized = false ;
    this.init = function () {
        if(that.is_initialized) return false ;
        this.is_initialized = true ;
        this.calib_init() ;
        return true ;
    } ;


    /* --------------
     *  Report error
     * --------------
     */
    function report_error (msg) {
        $('#popupdialogs').html(
            '<span class="ui-icon ui-icon-alert" style="float:left;"></span> '+msg
        ) ;
        $('#popupdialogs').dialog({
            resizable: false ,
            modal: true ,
            buttons: {
                Cancel: function () {
                    $(this).dialog('close') ;
                }
            } ,
            title: 'Error Message'
        }) ;
    } ;
}

var runtables = new runtables_create() ;
