#!/bin/sh
set -e

# Generate password hash for userlist.txt
echo "Generating password hash..."
PASSWORD_HASH=$(echo -n "pdfpassword${POSTGRES_USER}" | md5sum | awk '{print "md5"$1}')
echo "\"${POSTGRES_USER}\" \"${PASSWORD_HASH}\"" > /etc/pgbouncer/userlist.txt

echo "Starting PgBouncer..."
exec pgbouncer /etc/pgbouncer/pgbouncer.ini
