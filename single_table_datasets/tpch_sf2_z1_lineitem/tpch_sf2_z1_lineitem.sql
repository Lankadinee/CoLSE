CREATE TABLE tpch_sf2_z1 (
    l_orderkey  BIGINT,
    l_partkey  BIGINT,
    l_suppkey  BIGINT,
    l_linenumber  INTEGER,
    l_quantity  DOUBLE PRECISION,
    l_extendedprice  DOUBLE PRECISION,
    l_discount  DOUBLE PRECISION,
    l_tax  DOUBLE PRECISION,
    l_returnflag  CHAR(2),
    l_linestatus  CHAR(2),
    l_shipdate  VARCHAR(64),
    l_commitdate  VARCHAR(64),
    l_receiptdate  VARCHAR(64),
    l_shipinstruct  VARCHAR(64),
    l_shipmode  VARCHAR(64)
);