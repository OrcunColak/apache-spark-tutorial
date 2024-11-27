package com.colak;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;

public class FraudDetection {

    public static void main() {
        // Initialize SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("FraudDetection")
                .master("local[*]")
                .getOrCreate();

        // Load transactions from src/main/resources
        String filePath = FraudDetection.class.getClassLoader().getResource("transactions.csv").getPath();
        Dataset<Row> transactions = spark.read()
                .option("header", "true")
                .csv(filePath)
                .withColumn("amount", functions.col("amount").cast(DataTypes.DoubleType))
                .withColumn("timestamp", functions.col("timestamp").cast(DataTypes.TimestampType));

        // Criteria 1: Transactions exceeding $10,000
        Dataset<Row> highValueTransactions = transactions
                .filter(functions.col("amount").gt(10000))
                .withColumn("fraud_reason", functions.lit("High Value"));

        // Criteria 2: Transactions occurring too rapidly
        Dataset<Row> rapidTransactions = transactions
                .withColumn("previous_timestamp", functions.lag("timestamp", 1)
                        .over(Window.partitionBy("account_id").orderBy("timestamp")))
                .withColumn("time_diff", functions.unix_timestamp(functions.col("timestamp"))
                        .minus(functions.unix_timestamp(functions.col("previous_timestamp"))))
                .filter(functions.col("time_diff").lt(60))
                .withColumn("fraud_reason", functions.lit("Rapid Transactions"));

        // Criteria 3: International transactions exceeding $5,000
        Dataset<Row> internationalTransactions = transactions
                .filter(functions.col("is_international").equalTo("true")
                        .and(functions.col("amount").gt(5000)))
                .withColumn("fraud_reason", functions.lit("High Value International"));

        // Union all fraud cases
        Dataset<Row> flaggedTransactions = highValueTransactions
                .union(rapidTransactions)
                .union(internationalTransactions);

        // Show the flagged transactions
        flaggedTransactions.show();

        // Optionally, save flagged transactions to a database or file
        flaggedTransactions.write().format("csv").save("target/flagged_transactions");

        spark.stop();
    }
}

