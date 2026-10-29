-- requires: vector
-- Long-term customer memory: one summarized, embedded row per resolved ticket.
-- The dimension matches RELAY_EMBEDDING_DIM (default 256).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS customer_memory (
    customer_id  text NOT NULL,
    ticket_id    text NOT NULL,
    summary      text NOT NULL,
    resolved_at  timestamptz NOT NULL DEFAULT now(),
    embedding    vector(256) NOT NULL,
    PRIMARY KEY (customer_id, ticket_id)
);

CREATE INDEX IF NOT EXISTS customer_memory_embedding_idx
    ON customer_memory USING hnsw (embedding vector_cosine_ops);
