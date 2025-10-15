#!/bin/bash
# ============================================================================
# PostgreSQL Replication Initialization Script
# ============================================================================
# This script sets up the replication user and configuration on the primary
# database server to enable streaming replication to replica servers.
# ============================================================================

set -e

echo "Initializing PostgreSQL replication setup..."

# Configuration variables
REPLICATION_USER="${POSTGRES_REPLICATION_USER:-replicator}"
REPLICATION_PASSWORD="${POSTGRES_REPLICATION_PASSWORD:-replicator_pass}"

# Wait for PostgreSQL to be ready
until pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}"; do
  echo "Waiting for PostgreSQL to be ready..."
  sleep 2
done

echo "PostgreSQL is ready. Setting up replication..."

# Create replication user if it doesn't exist
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${REPLICATION_USER}') THEN
            CREATE ROLE ${REPLICATION_USER} WITH REPLICATION PASSWORD '${REPLICATION_PASSWORD}' LOGIN;
            RAISE NOTICE 'Replication user created: ${REPLICATION_USER}';
        ELSE
            RAISE NOTICE 'Replication user already exists: ${REPLICATION_USER}';
        END IF;
    END
    \$\$;

    -- Grant necessary permissions
    GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO ${REPLICATION_USER};
EOSQL

# Configure pg_hba.conf for replication
echo "Configuring pg_hba.conf for replication..."

PG_HBA_FILE="${PGDATA}/pg_hba.conf"

# Check if replication entry already exists
if ! grep -q "replication.*${REPLICATION_USER}" "$PG_HBA_FILE"; then
    echo "# Replication connections" >> "$PG_HBA_FILE"
    echo "host    replication     ${REPLICATION_USER}     0.0.0.0/0               md5" >> "$PG_HBA_FILE"
    echo "host    replication     ${REPLICATION_USER}     ::/0                    md5" >> "$PG_HBA_FILE"
    echo "Replication entries added to pg_hba.conf"
else
    echo "Replication entries already exist in pg_hba.conf"
fi

# Create replication slots for replicas (optional but recommended)
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    DO \$\$
    BEGIN
        -- Create replication slot for replica 1
        IF NOT EXISTS (SELECT 1 FROM pg_replication_slots WHERE slot_name = 'replica_1_slot') THEN
            PERFORM pg_create_physical_replication_slot('replica_1_slot');
            RAISE NOTICE 'Replication slot created: replica_1_slot';
        END IF;

        -- Create replication slot for replica 2
        IF NOT EXISTS (SELECT 1 FROM pg_replication_slots WHERE slot_name = 'replica_2_slot') THEN
            PERFORM pg_create_physical_replication_slot('replica_2_slot');
            RAISE NOTICE 'Replication slot created: replica_2_slot';
        END IF;
    END
    \$\$;
EOSQL

echo "Replication setup completed successfully!"
echo "Replication user: ${REPLICATION_USER}"
echo "Replicas can now connect using this user to stream WAL data."
