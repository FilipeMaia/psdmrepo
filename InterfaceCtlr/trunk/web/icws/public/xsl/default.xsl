<?xml version="1.0" encoding="ISO-8859-1"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

  <xsl:template name="domain">
    <script type="text/javascript">
      <xsl:text>
            function deleteParam(param, url, instrument, experiment)
            {
                if (confirm("Do you want to delete parameter \"" + param + "\"?")) {
                
                    url = url+"?_method=DELETE;instrument="+instrument+";experiment="+experiment;
                    var client = new XMLHttpRequest();
                    client.open("GET", url, false);
                    client.send();
                
                    if (client.status == 200) {
                        window.location.reload();
                    } else {
                        alert("Error deleting parameter: " + client.responseText);
                    }
                }
	    }

            function updateParam(form, url, instrument, experiment)
            {
                //get value of the parameter
                var value = form.parmvalue.value;

                url = url+encodeURI("?_method=PUT;value="+value+";instrument="+instrument+";experiment="+experiment)
                var client = new XMLHttpRequest();
                client.open("GET", url, false);
                client.send();
            
                if (client.status == 200) {
                    window.location.reload();
                } else {
                    alert("Error updating parameter: " + client.responseText);
                }
            }

            function createParam(form)
            {
                var fd = new FormData();
                fd.append("section", form.section.value);
                fd.append("param", form.parmname.value);
                fd.append("value", form.parmvalue.value);
                fd.append("type", form.parmtype.value);
                fd.append("instrument", form.instrument.value);
                fd.append("experiment", form.experiment.value);
                fd.append("description", form.description.value);
                
                var url = "config";
                var client = new XMLHttpRequest();
                client.open("POST", url, false);
                client.send(fd);
            
                if (client.status == 200) {
                    window.location.reload();
                } else {
                    alert("Error creating parameter: " + client.responseText);
                }
            }
            
            function stopController(cid, url)
            {
                if (confirm("Do you want to stop controller " + cid + "?")) {
                
                    var client = new XMLHttpRequest();
                    client.open("DELETE", url, false);
                    client.send();
                
                    if (client.status == 200) {
                        window.location.reload();
                        alert("Controller may need few seconds to stop, please reload \npage later if it does not stop immediately.");
                    } else {
                        alert("Error stopping controller: " + client.responseText);
                    }
                }
      }
            
        </xsl:text>
    </script>
  </xsl:template>

  <xsl:template name="style">
    <style type="text/css">
      <xsl:text>
        body { font-family:"Trebuchet MS", Arial, Helvetica, sans-serif; }
        thead th { 
          text-align: center; 
        }
        td, th {
          text-align: left; 
          border-color: black;
          padding: 3px 5px 3px 5px;
        }
        td.parmvalue {
          white-space:nowrap;
        }
        td.section {
          background-color: #ddc;
          font-weight: bold;
        }
        table thead {
          background-color: #998;
          color: white;
          font-weight: bold;
        }
        table {
          border-collapse:collapse; 
          margin-left: 2%; 
          width: 96% 
        }
        table a { cursor: pointer; text-decoration: none; color: black; }
        table a:focus { color: blue; }
        table a:hover { color: blue; }
        h1 {
          background-color: #998;
          color: white;
          text-align: center;
          width: 96%; margin-left: 2%; 
          padding: 4px 0px 4px 0px;
          border-radius: 0.2em;
          box-shadow: 4px 4px 4px #555;
        }
        h2 {
          background-color: #ccc;
          text-align: left;
          width: 50%; margin-left: 2%;
          padding: 5px 0px 5px 5px;
          border-radius: 0.2em;
          box-shadow: 4px 4px 4px #888;
        }
        form { display: inline; }
     
        input.parmvalue { width: 25em } 

        .newparm { width: 25em } 
     
        td.formleft { text-align: right; width: 30% }
     
     </xsl:text>
    </style>
  </xsl:template>


  <xsl:template match="controllers">
        <h1>Controller status</h1>
        <br />
        <table rules="all" frame="border" class="params">
          <thead>
            <tr>
              <th></th>
              <th>ID</th>
              <th>Host</th>
              <th>Status</th>
              <th>Instruments</th>
              <th>Started</th>
              <th>Log file</th>
            </tr>
          </thead>
          <tbody class="params">
            <xsl:for-each select="controller">
              <tr>
                <th>
                  <form>
                    <input type="button" value="Stop">
                      <xsl:attribute name="onclick">
                        stopController("<xsl:value-of select="@id" />", "<xsl:value-of select="@stop_url" />")
                      </xsl:attribute>
                    </input>
                  </form>
                </th>
                <th><xsl:value-of select="@id" /> <br/></th>
                <td><xsl:value-of select="@host" /></td>
                <td><xsl:value-of select="@status" /></td>
                <td><xsl:value-of select="@instruments" /></td>
                <td><xsl:value-of select="@started" /></td>
                <td><a><xsl:attribute name="href"><xsl:value-of select="@log_url" /></xsl:attribute><xsl:value-of select="@log" /></a></td>
              </tr>
            </xsl:for-each>
          </tbody>
        </table>
  </xsl:template>


  <!-- config-sections -->

  <xsl:template match="config-sections">
    <html>
      <head>
        <title>Controller configuration</title>
        <xsl:call-template name="domain"></xsl:call-template>
        <xsl:call-template name="style"></xsl:call-template>
      </head>
      <body>


        <!--  get the status of the system and insert it before configuration -->      
        <xsl:apply-templates select="document('../system.xml')"/>

        <br />
        <hr />
        <br />

      
        <h1>Controller configuration</h1>
        <br />
        <table rules="all" frame="border" class="params">
          <thead>
            <tr>
              <th>Parameter</th>
              <th>Value</th>
              <th>Type</th>
              <th>Instr.</th>
              <th>Exper.</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody class="params">
            <xsl:for-each select="config-section">
              <tr>
                <td colspan="6" class="section">
                  Section [<xsl:value-of select="@name" />]
                </td>
              </tr>
              <xsl:for-each select="config-param">
                <tr>
                  <th>
                    <xsl:value-of select="@param" />
                  </th>
                  <td class="parmvalue">
                    <form>
                      <input type="text" class="parmvalue" name="parmvalue" onkeypress="return event.keyCode!=13">
                        <xsl:attribute name="value"><xsl:value-of select="@value" /></xsl:attribute>
                      </input>
                      <input type="button" value="Save">
                        <xsl:attribute name="onclick">
                          updateParam(this.form,"<xsl:value-of select="@url" />",
                            "<xsl:value-of select="@instrument" />",
                            "<xsl:value-of select="@experiment" />")
                        </xsl:attribute>
                      </input>
                      <input type="button" value="Del">
                        <xsl:attribute name="onclick">
                          deleteParam("<xsl:value-of select="@param" />",
                            "<xsl:value-of select="@url" />",
                            "<xsl:value-of select="@instrument" />",
                            "<xsl:value-of select="@experiment" />")
                        </xsl:attribute>
                      </input>
                    </form>
                  </td>
                  <td>
                    <xsl:value-of select="@type" />
                  </td>
                  <td>
                    <xsl:value-of select="@instrument" />
                  </td>
                  <td>
                    <xsl:value-of select="@experiment" />
                  </td>
                  <td>
                    <xsl:value-of select="@description" />
                  </td>
                </tr>
              </xsl:for-each>
            </xsl:for-each>
          </tbody>
        </table>

        <br />
        <hr />
        <br />

        <h1>Create New Parameter</h1>

        <form>
          <table class="newparam">
            <tr>
              <td class="formleft">Section:</td>
              <td class="formright">
                <input type="text" class="newparm" name="section" onkeypress="return event.keyCode!=13" />
              </td>
            </tr>
            <tr>
              <td class="formleft">Parameter:</td>
              <td class="formright">
                <input type="text" class="newparm" name="parmname" size="16" onkeypress="return event.keyCode!=13" />
              </td>
            </tr>
            <tr>
              <td class="formleft">Value:</td>
              <td class="formright">
                <input type="text" class="newparm" name="parmvalue" onkeypress="return event.keyCode!=13" />
              </td>
            </tr>
            <tr>
              <td class="formleft">Type:</td>
              <td class="formright">
                <select name="parmtype" class="newparm">
                  <option value="String">String</option>
                  <option value="Integer">Integer</option>
                  <option value="Float">Float</option>
                  <option value="Date/Time">Date/Time</option>
                </select>
              </td>
            </tr>
            <tr>
              <td class="formleft">Instrument:</td>
              <td class="formright">
                <input type="text" class="newparm" name="instrument" size="4" onkeypress="return event.keyCode!=13" />
              </td>
            </tr>
            <tr>
              <td class="formleft">Experiment:</td>
              <td class="formright">
                <input type="text" class="newparm" name="experiment" size="6" onkeypress="return event.keyCode!=13" />
              </td>
            </tr>
            <tr>
              <td class="formleft">Description:</td>
              <td class="formright">
                <textarea class="newparm" name="description" rows="3" cols="50" onkeypress="return event.keyCode!=13" />
              </td>
            </tr>
            <tr>
              <td></td>
              <td>
                <input type="button" value="Create" onclick="createParam(this.form)" />
                <input type="reset" />
              </td>
            </tr>
          </table>
        </form>

      </body>
    </html>
  </xsl:template>

</xsl:stylesheet>
