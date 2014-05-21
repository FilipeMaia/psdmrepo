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
                this.update() ;
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

        this.wa = null ;    // work area container

        this.is_initialized = false ;

        this.init = function () {

            var that = this ;

            if (this.is_initialized) return ;
            this.is_initialized = true ;

            this.container.html('<div id="exp-info"></div>') ;
            this.wa = this.container.find('div#exp-info') ;

            if (!this.access_list.experiment.view_info) {
                this.wa.html(this.access_list.no_page_access_html) ;
                return ;
            }

            var html =
'<table><tbody>' +
'  <tr><td class="table_cell table_cell_left"  >Id</td>' +
'      <td class="table_cell table_cell_right" >'+this.experiment.id+'</td></tr>' ;
            if (this.experiment.is_facility) html +=
'  <tr><td class="table_cell table_cell_left"                   >Total # of e-Log entries</td>' +
'      <td class="table_cell table_cell_right" id="num_entries" >Loading...</td></tr>' +
'  <tr><td class="table_cell table_cell_left"                   >Last entry</td>' +
'      <td class="table_cell table_cell_right" id="last_entry"  >Loading...</td></tr>' ;
            else html +=
'  <tr><td class="table_cell table_cell_left"                   >Status</td>' +
'      <td class="table_cell table_cell_right" id="status"      >Loading...</td></tr>' +
'  <tr><td class="table_cell table_cell_left"                                 >Total # of runs taken</td>' +
'      <td class="table_cell table_cell_right" id="num_runs"                  >Loading...</td></tr>' +
'  <tr><td class="table_cell table_cell_left"                                 >First run</td>' +
'      <td class="table_cell table_cell_right" id="first_run"                 >Loading...</td></tr>' +
'  <tr><td class="table_cell table_cell_left"                                 >Last run</td>' +
'      <td class="table_cell table_cell_right" id="last_run"                  >Loading...</td></tr>' ;
         html +=
'  <tr><td class="table_cell table_cell_left"                                 >Description</td>' +
'      <td class="table_cell table_cell_right" id="description"               ><div class="exp-info-descr">Loading...</div></td></tr>' +
'  <tr><td class="table_cell table_cell_left"                                 >Contact person(s)</td>' +
'      <td class="table_cell table_cell_right" id="contact"                   >Loading...</td></tr>' +
'  <tr><td class="table_cell table_cell_left"                                 >UNIX Account of PI</td>' +
'      <td class="table_cell table_cell_right"                                >'+this.experiment.contact_uid+'</td></tr>' +
'  <tr><td class="table_cell table_cell_left  table_cell_bottom" valign="top" >Experiment Group</td>' +
'      <td class="table_cell table_cell_right table_cell_bottom"              >' +
'        <table cellspacing="0" cellpadding="0"><tbody>' +
'          <tr><td valign="top">'+this.experiment.posix_group+'</td>' +
'              <td>&nbsp;</td>' +
'              <td><span class="toggler ui-icon ui-icon-triangle-1-e" id="group-toggler" title="click to see/hide the list of members"></span>' +
'                  <div  class="group-hidden"                         id="group-members" >Loading...</div>' +
'              </td></tr>' +
'        </tbody></table>' +
'      </td></tr>' +
'</tbody></table>' ;
            this.wa.html(html) ;

            this.group_toggler = this.wa.find('#group-toggler') ;
            this.group_members = this.wa.find('#group-members') ;

            this.group_toggler.click(function () {
                if (that.group_members.hasClass('group-hidden')) {
                    that.group_members.removeClass('group-hidden').addClass('group-visible') ;
                    that.group_toggler.removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s') ;
                } else {
                    that.group_members.removeClass('group-visible').addClass('group-hidden') ;
                    that.group_toggler.removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e') ;
                }
            });
        } ;

        this.updated_once = false ;
        this.update = function () {
            if (!this.access_list.experiment.view_info) return ;
            if (this.updated_once) return ;
            this.updated_once = true ;
            this.load() ;
        } ;

        this.load = function () {
            var that = this ;
            Fwk.web_service_GET (
                '../portal/ws/experiment_info.php', {id:this.experiment.id} ,
                function (data) {
                    if (data.status === 'success') that.display(data) ;
                }   
            ) ;
        } ;
        this.display = function (data) {
            if (this.experiment.is_facility) {
                this.wa.find('#num_entries').html(data.num_elog_entries) ;
                this.wa.find('#last_entry' ).html(data.last_elog_entry_posted) ;
            } else {
                this.wa.find('#status').html(data.is_active ?
'<span style="color:#ff0000; font-weight:bold;" >ACTIVE</span>' :
'<span style="color:#b0b0b0; font-weight:bold;" >NOT ACTIVE</span>'
                ) ;
                this.wa.find('#num_runs'   ).html(data.num_runs) ;
                this.wa.find('#first_run'  ).html(data.first_run.num ? data.first_run.begin_time+' (<b>run '+data.first_run.num+'</b>)' : 'n/a') ;
                this.wa.find('#last_run'   ).html(data.last_run.num  ? data.last_run.begin_time +' (<b>run '+data.last_run.num +'</b>)' : 'n/a') ;
            }
            this.wa.find('#description').html(data.description) ;
            this.wa.find('#contact'    ).html(data.contact_info_decorated) ;
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
            this.group_members.html(html) ;
        } ;
    }
    Class.define_class (Experiment_Info, FwkApplication, {}, {}) ;

    return Experiment_Info ;
}) ;