#!/bin/bash
# Creates the second database used by the LangGraph backend. Runs once on first
# container start (docker-entrypoint-initdb.d).
set -e
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" <<-SQL
  SELECT 'CREATE DATABASE agentmem_lg'
  WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'agentmem_lg')\gexec
SQL
