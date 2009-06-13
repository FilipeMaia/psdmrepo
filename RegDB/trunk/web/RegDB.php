<!--
To change this template, choose Tools | Templates
and open the template in the editor.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
  <head>
    <title>LogBook PHP API Tests</title>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  </head>
  <body>
    <style>
      .category {
        color:#0071bc;
      }
      .actions_menu {
        margin-left:2em;
        border-right-style:solid;
        border-right-width:1px;
        width:20em;
      }
      .actions {
        margin-left:2em;
      }
      .instructions {
          margin-left:2em;
          margin-right:2em;
      }
    </style>
    <br>
    <h1><center>Registration Database PHP API Tests</center></h1>
    <br>
    <hr style="margin-left:1em; margin-right:1em; margin-bottom:1em;">
    <table>
      <tbody>
        <tr>
          <td>
            <div class="actions_menu">
              <h2 class="category">Experiments</h2>
              <div class="actions">
                <ul>
                  <li><a href="ListExperiments.php">List all</a></li>
                  <li><a href="CreateExperiment.php">Register new experiment</a></li>
                  <li><a href="CloseExperiment.php">Close open-ended experiment</a></li>
                </ul>
              </div>
              <h2 class="category">Shifts</h2>
              <div class="actions">
                <ul>
                  <li><a href="ListShifts.php">List all</a></li>
                  <li><a href="CreateShift.php">Begin new shift</a></li>
                  <li><a href="CloseShift.php">Close open-ended shift</a></li>
                </ul>
              </div>
              <h2 class="category">Runs</h2>
              <div class="actions">
                <ul>
                  <li><a href="ListRuns.php">List all</a></li>
                  <li><a href="CreateRun.php">Begin new run</a></li>
                  <li><a href="CloseRun.php">Close open-ended run</a></li>
                </ul>
              </div>
              <h2 class="category">Run Summary Parameters</h2>
              <div class="actions">
                <ul>
                  <li><a href="ListRunParams.php">List all</a></li>
                  <li><a href="CreateRunParam.php">Create new parameter</a></li>
                  <li><a href="SetRunParamValue.php">Set parameter's value</a></li>
                  <li><a href="ListRunParamValues.php">List values of parameters</a></li>
                </ul>
              </div>
              <h2 class="category">Free-form Entries</h2>
              <div class="actions">
                <ul>
                  <li><a href="ListFFEntries.php">List all</a></li>
                  <li><a href="CreateFFEntry.php">Create an entry</a></li>
                  <li><a href="AddTags.php">Add tag(s) for an entry</a></li>
                  <li><a href="AttachDocuments.php">Attach document(s) to an entry</a></li>
                </ul>
              </div>
              <h2 class="category">More</h2>
              <div class="actions">
                <ul>
                  <li><a href="CurrentStatus.php">Display current status</a></li>
                </ul>
              </div>
            </div>
          </td>
          <td valign="top">
            <div class="instructions">
            <p>This document presents a collection of low-level tests for the LogBook PHP API.
            The tests are groupped into four major categories. Each link under the categories
            corresponds to an atomic operations to be preformed via the API. Links will
            follow to separate pages.</p>
            </div>
          </td>
        </tr>
      </tbody>
    </table>
  </body>
</html>
