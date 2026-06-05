"""Capability 5 — the SAME batch aggregates as analytics.py, written as a PySpark
DataFrame pipeline (BAX-423 Lecture 4: PySpark, group-by, window functions, Parquet).

This is the distributed-framework path. The shipped app runs analytics.py (pandas),
because the demo host has no JVM and the spec accepts "a distributed OR batch
framework"; this module is the drop-in Spark equivalent for a cluster, and it RUNS
wherever Java + pyspark are installed:

    pip install pyspark            # needs a JRE/JDK on PATH
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=code .venv/bin/python code/analytics_spark.py

It demonstrates the L4 techniques on the real corpus:
  - DataFrame group-by/agg     -> category distribution, active communities
  - Window + lag               -> week-over-week share momentum
  - Parquet columnar write     -> reports/spark/ (partitioned, splittable)

The aggregates are identical in meaning to analytics.py; only the engine differs.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "spark"


def main() -> None:
    try:
        from pyspark.sql import SparkSession, Window
        from pyspark.sql import functions as F
    except ImportError:
        print("pyspark is not installed (and needs a JRE). This is the cluster path; the "
              "shipped snapshot was built with the equivalent pandas job, analytics.py.")
        sys.exit(0)

    from engageiq.embed import DB_PATH

    spark = (SparkSession.builder.appName("EngageIQ-Analytics")
             .master("local[*]").config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    # Load the corpus (pandas read -> Spark DataFrame, so no JDBC driver is needed).
    conn = sqlite3.connect(str(DB_PATH))
    pdf = pd.read_sql_query(
        "SELECT opportunity_id, source, domain, created_at, author, community, "
        "score, num_comments FROM opportunities", conn)
    conn.close()
    pdf["engagement"] = pdf["score"].fillna(0).astype(int) + pdf["num_comments"].fillna(0).astype(int)
    df = spark.createDataFrame(pdf)
    df = df.withColumn("week", F.date_format(F.date_trunc("week", F.to_timestamp("created_at")), "yyyy-MM-dd"))
    df = df.filter(F.col("domain").isNotNull() & (F.col("domain") != "")).cache()
    print(f"[spark] loaded {df.count()} opportunities")

    # 1. category distribution (group-by) ---------------------------------------
    dist = (df.groupBy("domain")
            .agg(F.count("*").alias("volume"),
                 F.countDistinct("author").alias("authors"),
                 F.round(F.avg("engagement"), 1).alias("avg_engagement"))
            .orderBy(F.desc("volume")))
    print("[spark] category distribution:")
    dist.show(20, truncate=False)

    # 2. most active communities (group-by + filter) ----------------------------
    comm = (df.filter(F.col("community").isNotNull())
            .groupBy("domain", "community")
            .agg(F.sum("engagement").alias("engagement"), F.count("*").alias("volume"))
            .orderBy(F.desc("engagement")))

    # 3. week-over-week SHARE momentum (window + lag) ----------------------------
    weekly = df.groupBy("domain", "week").agg(F.count("*").alias("c"))
    totals = df.groupBy("week").agg(F.count("*").alias("total"))
    weekly = weekly.join(totals, "week").withColumn("share", F.col("c") / F.col("total"))
    w = Window.partitionBy("domain").orderBy("week")
    momentum = (weekly.withColumn("prev_share", F.lag("share").over(w))
                .withColumn("delta_pp", F.round((F.col("share") - F.col("prev_share")) * 100, 2)))
    print("[spark] latest week-over-week share momentum (sample):")
    (momentum.orderBy(F.desc("week"), F.desc("delta_pp"))
     .select("week", "domain", "share", "delta_pp").show(15, truncate=False))

    # 4. write Parquet (columnar, splittable) -----------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    dist.write.mode("overwrite").parquet(str(OUT / "category_distribution"))
    comm.write.mode("overwrite").parquet(str(OUT / "active_communities"))
    momentum.write.mode("overwrite").parquet(str(OUT / "weekly_share_momentum"))
    print(f"[spark] wrote Parquet outputs under {OUT}")
    spark.stop()


if __name__ == "__main__":
    main()
