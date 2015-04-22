




Mover modes
+++++++++++

A mover-mode selects the options that are used for bbcp to transfer a file.
The main differentiation is if the file is transferred from a locally mounted file
system or a remote server and it the transfer is in realtime.

+-------------+--------------+----------------+-----------+----------+
| mode        | transfer     | Example        | Real time | instr in |
|             |              |                |           | trg path |
+=============================================+===========+----------+
| dss-ffb     | remote-local | dss-/brick1/   | yes       | no       |
+------------------------+------------+-------+-----------+----------|
| ffb-offline | remote-local | ffb-Lustre     | no        | yes      |
+------------------------+------------+-------+-----------+----------|
| ffb-local   | local-local  | Gluster-Lustre | no        | yes      |
+---------------------------------------------+-----------+----------|
| ioc-ffb     | local-local  | nfs-Gluster    | yes       | yes      |
|             |              | nfs-Lustre     |           |          |
+---------------------------------------------+-----------+----------|
| local       | local-local  | nfs-Lustre     + yes       | yes      |
+-------------+--------------+----------------+-----------+----------|
