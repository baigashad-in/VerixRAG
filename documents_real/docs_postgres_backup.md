<!-- Source: https://raw.githubusercontent.com/postgres/postgres/master/doc/src/sgml/backup.sgml -->
<!-- Domain: raw.githubusercontent.com -->

<!-- doc/src/sgml/backup.sgml -->

<chapter id="backup">
 <title>Backup and Restore</title>

 <indexterm zone="backup"><primary>backup</primary></indexterm>

 <para>
  As with everything that contains valuable data, <productname>PostgreSQL</productname>
  databases should be backed up regularly. While the procedure is
  essentially simple, it is important to have a clear understanding of
  the underlying techniques and assumptions.
 </para>

 <para>
  There are three fundamentally different approaches to backing up
  <productname>PostgreSQL</productname> data:
  <itemizedlist>
   <listitem><para><acronym>SQL</acronym> dump</para></listitem>
   <listitem><para>File system level backup</para></listitem>
   <listitem><para>Continuous archiving</para></listitem>
  </itemizedlist>
  Each has its own strengths and weaknesses; each is discussed in turn
  in the following sections.
 </para>

 <sect1 id="backup-dump">
  <title><acronym>SQL</acronym> Dump</title>

  <para>
   The idea behind this dump method is to generate a file with SQL
   commands that, when fed back to the server, will recreate the
   database in the same state as it was at the time of the dump.
   <productname>PostgreSQL</productname> provides the utility program
   <xref linkend="app-pgdump"/> for this purpose. The basic usage of this
   command is:
<synopsis>
pg_dump <replaceable class="parameter">dbname</replaceable> &gt; <replaceable class="parameter">dumpfile</replaceable>
</synopsis>
   As you see, <application>pg_dump</application> writes its result to the
   standard output. We will see below how this can be useful.
   While the above command creates a text file, <application>pg_dump</application>
   can create files in other formats that allow for parallelism and more
   fine-grained control of object restoration.
  </para>

  <para>
   <application>pg_dump</application> is a regular <productname>PostgreSQL</productname>
   client application (albeit a particularly clever one). This means
   that you can perform this backup procedure from any remote host that has
   access to the database. But remember that <application>pg_dump</application>
   does not operate with special permissions. In particular, it must
   have read access to all tables that you want to back up, so in order
   to back up the entire database you almost always have to run it as a
   database superuser.  (If you do not have sufficient privileges to back up
   the entire database, you can still back up portions of the database to which
   you do have access using options such as
   <option>-n <replaceable>schema</replaceable></option>
   or <option>-t <replaceable>table</replaceable></option>.)
  </para>

  <para>
   To specify which database server <application>pg_dump</application> should
   contact, use the command line options <option>-h
   <replaceable>host</replaceable></option> and <option>-p <replaceable>port</replaceable></option>. The
   default host is the local host or whatever your
   <envar>PGHOST</envar> environment variable specifies. Similarly,
   the default port is indicated by the <envar>PGPORT</envar>
   environment variable or, failing that, by the compiled-in default.
   (Conveniently, the server will normally have the same compiled-in
   default.)
  </para>

  <para>
   Like any other <productname>PostgreSQL</productname> client application,
   <application>pg_dump</application> will by default connect with the database
   user name that is equal to the current operating system user name. To override
   this, either specify the <option>-U</option> option or set the
   environment variable <envar>PGUSER</envar>. Remember that
   <application>pg_dump</application> connections are subject to the normal
   client authentication mechanisms (which are described in <xref
   linkend="client-authentication"/>).
  </para>

  <para>
   An important advantage of <application>pg_dump</application> over the other backup
   methods described later is that <application>pg_dump</application>'s output can
   generally be re-loaded into newer versions of <productname>PostgreSQL</productname>,
   whereas file-level backups and continuous archiving are both extremely
   server-version-specific.  <application>pg_dump</application> is also the only method
   that will work when transferring a database to a different machine
   architecture, such as going from a 32-bit to a 64-bit server.
  </para>

  <para>
   Dumps created by <application>pg_dump</application> are internally consistent,
   meaning, the dump represents a snapshot of the database at the time
   <application>pg_dump</application> began running. <application>pg_dump</application> does not
   block other operations on the database while it is working.
   (Exceptions are those operations that need to operate with an
   exclusive lock, such as most forms of <command>ALTER TABLE</command>.)
  </para>

  <sect2 id="backup-dump-restore">
   <title>Restoring the Dump</title>

   <para>
    Text files created by <application>pg_dump</application> are intended to
    be read by the <application>psql</application> program using its default
    settings. The general command form to restore a text dump is
<synopsis>
psql -X <replaceable class="parameter">dbname</replaceable> &lt; <replaceable class="parameter">dumpfile</replaceable>
</synopsis>
    where <replaceable class="parameter">dumpfile</replaceable> is the
    file output by the <application>pg_dump</application> command. The database <replaceable
    class="parameter">dbname</replaceable> will not be created by this
    command, so you must create it yourself from <literal>template0</literal>
    before executing <application>psql</application> (e.g., with
    <literal>createdb -T template0 <replaceable
    class="parameter">dbname</replaceable></literal>).
    To ensure <application>psql</application> runs with its default settings,
    use the <option>-X</option> (<option>--no-psqlrc</option>) option.
    <application>psql</application>
    supports options similar to <application>pg_dump</application> for specifying
    the database server to connect to and the user name to use. See
    the <xref linkend="app-psql"/> reference page for more information.
   </para>

   <para>
    Non-text file dumps should be restored using the <xref
    linkend="app-pgrestore"/> utility.
   </para>

   <para>
    Before restoring an SQL dump, all the users who own objects or were
    granted permissions on objects in the dumped database must already
    exist. If they do not, the restore will fail to recreate the
    objects with the original ownership and/or permissions.
    (Sometimes this is what you want, but usually it is not.)
   </para>

   <para>
    By default, the <application>psql</application> script will continue to
    execute after an SQL error is encountered. You might wish to run
    <application>psql</application> with
    the <literal>ON_ERROR_STOP</literal> variable set to alter that
    behavior and have <application>psql</application> exit with an
    exit status of 3 if an SQL error occurs:
<programlisting>
psql -X --set ON_ERROR_STOP=on <replaceable>dbname</replaceable> &lt; <replaceable>dumpfile</replaceable>
</programlisting>
    Either way, you will only have a partially restored database.
    Alternatively, you can specify that the whole dump should be
    restored as a single transaction, so the restore is either fully
    completed or fully rolled back. This mode can be specified by
    passing the <option>-1</option> or <option>--single-transaction</option>
    command-line options to <application>psql</application>. When using this
    mode, be aware that even a minor error can rollback a
    restore that has already run for many hours. However, that might
    still be preferable to manually cleaning up a complex database
    after a partially restored dump.
   </para>

   <para>
    The ability of <application>pg_dump</application> and <application>psql</application> to
    write to or read from pipes makes it possible to dump a database
    directly from one server to another, for example:
<programlisting>
pg_dump -h <replaceable>host1</replaceable> <replaceable>dbname</replaceable> | psql -X -h <replaceable>host2</replaceable> <replaceable>dbname</replaceable>
</programlisting>
   </para>

   <important>
    <para>
     The dumps produced by <application>pg_dump</application> are relative to
     <literal>template0</literal>. This means that any languages, procedures,
     etc. added via <literal>template1</literal> will also be dumped by
     <application>pg_dump</application>. As a result, when restoring, if you are
     using a customized <literal>template1</literal>, you must create the
     empty database from <literal>template0</literal>, as in the example
     above.
    </para>
   </important>

   <para>
    After restoring a backup, it is wise to run <link
    linkend="sql-analyze"><command>ANALYZE</command></link> on each
    database so the query optimizer has useful statistics;
    see <xref linkend="vacuum-for-statistics"/>
    and <xref linkend="autovacuum"/> for more information.
    For more advice on how to load large amounts of data
    into <productname>PostgreSQL</productname> efficiently, refer to <xref
    linkend="populate"/>.
   </para>
  </sect2>

  <sect2 id="backup-dump-all">
   <title>Using <application>pg_dumpall</application></title>

   <para>
    <application>pg_dump</application> dumps only a single database at a time,
    and it does not dump information about roles or tablespaces
    (because those are cluster-wide rather than per-database).
    To support convenient dumping of the entire contents of a database
    cluster, the <xref linkend="app-pg-dumpall"/> program is provided.
    <application>pg_dumpall</application> backs up each database in a given
    cluster, and also preserves cluster-wide data such as role and
    tablespace definitions. The basic usage of this command is:
<synopsis>
pg_dumpall &gt; <replaceable>dumpfile</replaceable>
</synopsis>
    The resulting dump can be restored with <application>psql</application>:
<synopsis>
psql -X -f <replaceable class="parameter">dumpfile</replaceable> postgres
</synopsis>
    (Actually, you can specify any existing database name to start from,
    but if you are loading into an empty cluster then <literal>postgres</literal>
    should usually be used.)  It is always necessary to have
    database superuser access when restoring a <application>pg_dumpall</application>
    dump, as that is required to restore the role and tablespace information.
    If you use tablespaces, make sure that the tablespace paths in the
    dump are appropriate for the new installation.
   </para>

   <para>
    <application>pg_dumpall</application> works by emitting commands to re-create
    roles, tablespaces, and empty databases, then invoking
    <application>pg_dump</application> for each database.  This means that while
    each database will be internally consistent, the snapshots of
    different databases are not synchronized.
   </para>

   <para>
    Cluster-wide data can be dumped alone using the
    <application>pg_dumpall</application> <option>--globals-only</option> option.
    This is necessary to fully backup the cluster if running the
    <application>pg_dump</application> command on individual databases.
   </para>
  </sect2>

  <sect2 id="backup-dump-large">
   <title>Handling Large Databases</title>

   <para>
    Some operating systems have maximum file size limits that cause
    problems when creating large <application>pg_dump</application> output files.
    Fortunately, <application>pg_dump</application> can write to the standard
    output, so you can use standard Unix tools to work around this
    potential problem.  There are several possible methods:
   </para>

   <formalpara>
    <title>Use compressed dumps.</title>
    <para>
     You can use your favorite compression program, for example
     <application>gzip</application>:

<programlisting>
pg_dump <replaceable class="parameter">dbname</replaceable> | gzip &gt; <replaceable class="parameter">filename</replaceable>.gz
</programlisting>

     Reload with:

<programlisting>
gunzip -c <replaceable class="parameter">filename</replaceable>.gz | psql <replaceable class="parameter">dbname</replaceable>
</programlisting>

     or:

<programlisting>
cat <replaceable class="parameter">filename</replaceable>.gz | gunzip | psql <replaceable class="parameter">dbname</replaceable>
</programlisting>
    </para>
   </formalpara>

   <formalpara>
    <title>Use <command>split</command>.</title>
    <para>
     The <command>split</command> command
     allows you to split the output into smaller files that are
     acceptable in size to the underlying file system. For example, to
     make 2 gigabyte chunks:

<programlisting>
pg_dump <replaceable class="parameter">dbname</replaceable> | split -b 2G - <replaceable class="parameter">filename</replaceable>
</programlisting>

     Reload with:

<programlisting>
cat <replaceable class="parameter">filename</replaceable>* | psql <replaceable class="parameter">dbname</replaceable>
</programlisting>

     If using GNU <application>split</application>, it is possible to
     use it and <application>gzip</application> together:

<programlisting>
pg_dump <replaceable class="parameter">dbname</replaceable> | split -b 2G --filter='gzip > $FILE.gz'
</programlisting>

     It can be restored using <command>zcat</command>.
    </para>
   </formalpara>

   <formalpara>
    <title>Use <application>pg_dump</application>'s custom dump format.</title>
    <para>
     If <productname>PostgreSQL</productname> was built on a system with the
     <application>zlib</application> compression library installed, the custom dump
     format will compress data as it writes it to the output file. This will
     produce dump file sizes similar to using <command>gzip</command>, but it
     has the added advantage that tables can be restored selectively. The
     following command dumps a database using the custom dump format:

<programlisting>
pg_dump -Fc <replaceable class="parameter">dbname</replaceable> &gt; <replaceable class="parameter">filename</replaceable>
</programlisting>

     A custom-format dump is not a script for <application>psql</application>, but
     instead must be restored with <application>pg_restore</application>, for example:

<programlisting>
pg_restore -d <replaceable class="parameter">dbname</replaceable> <replaceable class="parameter">filename</replaceable>
</programlisting>

     See the <xref linkend="app-pgdump"/> and <xref
     linkend="app-pgrestore"/> reference pages for details.
    </para>
   </formalpara>

   <para>
    For very large databases, you might need to combine <command>split</command>
    with one of the other two approaches.
   </para>

   <formalpara>
    <title>Use <application>pg_dump</application>'s parallel dump feature.</title>
    <para

[Document truncated for evaluation purposes]