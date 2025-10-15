#!/bin/bash
set -e

echo "Initializing PostgreSQL replica..."

# Check if data directory is already initialized
if [ -s "$PGDATA/PG_VERSION" ]; then
    echo "Data directory already initialized, skipping pg_basebackup"
else
    echo "Data directory empty, performing base backup from primary..."

    # Wait for primary to be ready
    until pg_isready -h ${POSTGRES_PRIMARY_HOST} -U ${POSTGRES_USER}; do
        echo "Waiting for primary database..."
        sleep 2
    done

    echo "Primary is ready, starting base backup..."

    # Create pgpass file to avoid password prompts
    echo "${POSTGRES_PRIMARY_HOST}:5432:*:${POSTGRES_REPLICATION_USER}:${POSTGRES_REPLICATION_PASSWORD}" > ~/.pgpass
    chmod 600 ~/.pgpass

    # Perform base backup WITHOUT -W flag
    PGPASSFILE=~/.pgpass pg_basebackup -h ${POSTGRES_PRIMARY_HOST} \
        -D ${PGDATA} \
        -U ${POSTGRES_REPLICATION_USER} \
        -v \
        -P \
        --wal-method=stream

    # Configure replica
    echo "primary_conninfo = 'host=${POSTGRES_PRIMARY_HOST} port=5432 user=${POSTGRES_REPLICATION_USER} password=${POSTGRES_REPLICATION_PASSWORD}'" >> ${PGDATA}/postgresql.auto.conf
    touch ${PGDATA}/standby.signal

    echo "Base backup complete!"
fi

# Start PostgreSQL in hot standby mode
exec postgres -c hot_standby=on
