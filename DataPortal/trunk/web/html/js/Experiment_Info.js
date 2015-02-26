define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk) {

    cssloader.load('../portal/css/Experiment_Info.css') ;

    /**
     * The application for displaying the general information about the experiment
     *
     * @returns {Experiment_Info}
     */
    function Experiment_Info (experiment, access_list) {

        var _that = this ;

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
            this._init() ;
        } ;

        this._update_interval_sec = 30 ;
        this._prev_update_sec = null ;

        this.on_update = function () {
            if (this.active) {
                this._init() ;
                var now_sec = Fwk.now().sec ;
                if (!this._prev_update_sec || (now_sec - this._prev_update_sec) > this._update_interval_sec) {
                    this._prev_update_sec = now_sec ;
                    this._load() ;
                }
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

        this._is_initialized = false ;

        this._wa = function (html) {
            if (!this._wa_elem) {
                var this_html = html ||
'<div id="exp-info" >' +

  '<div class="info" id="updated" style="float:right;" ></div> ' +
  '<div style="clear:both;" ></div> ' +

  '<div id="table" style="float:left;" > ' +
    '<table><tbody> ' +
      '<tr><td class="table_cell table_cell_left"  >Id</td> ' +
          '<td class="table_cell table_cell_right" >'+this.experiment.id+'</td></tr> ' +
                    (this.experiment.is_facility ?
    '<tr><td class="table_cell table_cell_left"                   >Total # of e-Log entries</td> ' +
        '<td class="table_cell table_cell_right" id="num_entries" >Loading...</td></tr> ' +
    '<tr><td class="table_cell table_cell_left"                   >Last entry</td> ' +
        '<td class="table_cell table_cell_right" id="last_entry"  >Loading...</td></tr> '
                    :
    '<tr><td class="table_cell table_cell_left"                   >Status</td> ' +
        '<td class="table_cell table_cell_right" id="status"      >Loading...</td></tr> ' +
    '<tr><td class="table_cell table_cell_left"                                 >Total # of runs taken</td> ' +
        '<td class="table_cell table_cell_right" id="num_runs"                  >Loading...</td></tr> ' +
    '<tr><td class="table_cell table_cell_left"                                 >First run</td> ' +
        '<td class="table_cell table_cell_right" id="first_run"                 >Loading...</td></tr> ' +
    '<tr><td class="table_cell table_cell_left"                                 >Last run</td> ' +
        '<td class="table_cell table_cell_right" id="last_run"                  >Loading...</td></tr> '
                    ) +
    '<tr><td class="table_cell table_cell_left"                    valign="top" >Description</td> ' +
        '<td class="table_cell table_cell_right" id="description"               ><div class="exp-info-descr">Loading...</div></td></tr> ' +
    '<tr><td class="table_cell table_cell_left"                                 >Contact person(s)</td> ' +
        '<td class="table_cell table_cell_right" id="contact"                   >Loading...</td></tr> ' +
    '<tr><td class="table_cell table_cell_left"                                 >UNIX Account of PI</td> ' +
        '<td class="table_cell table_cell_right"                                >'+this.experiment.contact_uid+'</td></tr> ' +
    '<tr><td class="table_cell table_cell_left  table_cell_bottom" valign="top" >Experiment Group</td> ' +
        '<td class="table_cell table_cell_right table_cell_bottom"              > ' +
          '<table cellspacing="0" cellpadding="0"><tbody> ' +
            '<tr><td valign="top">'+this.experiment.posix_group+'</td> ' +
                '<td>&nbsp;</td> ' +
                '<td><span class="toggler ui-icon ui-icon-triangle-1-e" id="group-toggler" title="click to see/hide the list of members"></span> ' +
                    '<div  class="group-hidden"                         id="group-members" >Loading...</div> ' +
                '</td></tr> ' +
          '</tbody></table> ' +
        '</td></tr> ' +
  '</tbody></table> ' +
  '</div> ' +
  '<div id="buttons" style="float:left;" > ' +
    '<button class="control-button" name="update" title="click to update the summary information"><img src="../webfwk/img/Update.png" /></button> ' +
  '</div> ' +
  '<div style="clear:both;" ></div> ' +
'</div>' ;
                this.container.html(this_html) ;
                this._wa_elem = this.container.children('#exp-info') ;
            }
            return this._wa_elem ;
        } ;
        this._set_updated = function (html) {
            if (!this._updated_elem) this._updated_elem = this._wa().children('#updated') ;
            this._updated_elem.html(html) ;
        } ;
        this._button_update = function () {
            if (!this._button_update_elem) {
                this._button_update_elem = this._wa().find('button[name="update"]').button() ;
            }
            return this._button_update_elem ;
        } ;
        this._toggler = function () {
            if (!this._toggler_elem) {
                this._toggler_elem = this._wa().find('#group-toggler') ;
            }
            return this._toggler_elem ;
        } ;
        this._members = function () {
            if (!this._members_elem) {
                this._members_elem = this._wa().find('#group-members') ;
            }
            return this._members_elem ;
        } ;


        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            if (!this.access_list.experiment.view_info) {
                this._wa(this.access_list.no_page_access_html) ;
                return ;
            }

            this._button_update().click(function () { _that._load() ; }) ;

            this._toggler().click(function () {
                if (_that._members().hasClass('group-hidden')) {
                    _that._members().removeClass('group-hidden').addClass('group-visible') ;
                    _that._toggler().removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s') ;
                } else {
                    _that._members().removeClass('group-visible').addClass('group-hidden') ;
                    _that._toggler().removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e') ;
                }
            });
        } ;
        this._load = function () {
            this._set_updated('Updating...') ;
            Fwk.web_service_GET (
                '../portal/ws/experiment_info.php', {id:this.experiment.id} ,
                function (data) {
                    _that._set_updated('Updated: <b>'+data.updated+'</b>') ;
                    _that._display(data) ;
                }   
            ) ;
        } ;
        this._display = function (data) {
            if (this.experiment.is_facility) {
                this._wa().find('#num_entries').html(data.num_elog_entries) ;
                this._wa().find('#last_entry' ).html(data.last_elog_entry_posted) ;
            } else {
                this._wa().find('#status').html(data.is_active ?
'<span style="color:#ff0000; font-weight:bold;" >ACTIVE</span>' :
'<span style="color:#b0b0b0; font-weight:bold;" >NOT ACTIVE</span>'
                ) ;
                this._wa().find('#num_runs'   ).html(data.num_runs) ;
                this._wa().find('#first_run'  ).html(data.first_run.num ? data.first_run.begin_time+' (<b>run '+data.first_run.num+'</b>)' : 'n/a') ;
                this._wa().find('#last_run'   ).html(data.last_run.num  ? data.last_run.begin_time +' (<b>run '+data.last_run.num +'</b>)' : 'n/a') ;
            }
            this._wa().find('#description').children('.exp-info-descr').html(data.description) ;
            this._wa().find('#contact'    )                            .html(data.contact_info_decorated) ;
            var html =
'<table><tbody>' +
'  <tr><td class="table_cell table_cell_left"  ></td>' +
'      <td class="table_cell table_cell_right" ></td></tr>' ;
            for (var i in data.group_members) {
                var member = data.group_members[i] ;
                html +=
'  <tr><td class="table_cell table_cell_left"  >'+member.uid+'</td>' +
'      <td class="table_cell table_cell_right" >'+member.gecos+'</td></tr>' ;
            }
            html +=
'</tbody></table>' ;
            this._members().html(html) ;
        } ;
    }
    Class.define_class (Experiment_Info, FwkApplication, {}, {}) ;

    return Experiment_Info ;
}) ;