from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from functools import reduce
from pyspark.sql.types import *
import pandas as pd

class SparkDataCheck:
    def __init__(self, df: DataFrame):
        self.df = df
        
    # Create classmethod to read in spark format csv    
    @classmethod
    def from_csv(cls, spark, path):
        df = spark.read.load(path, format="csv", header=True, inferSchema=True)
        return cls(df)

    # Create classmethod to read in pandas format csv
    @classmethod
    def from_pandas(cls, spark, pandas_df):
        df = spark.createDataFrame(pandas_df)
        return cls(df)
    
    # Method that checks cols for numeric values
    def check_numeric_range(self, column, lower=None, upper=None):

        # Check if at least one bound is provided
        if lower is None and upper is None:
            print("Error: Missing a lower OR upper bound.")
            return self

        # Check column type
        dtype_dict = dict(self.df.dtypes)

        numeric_types = ["int", "bigint", "double", "float", "long", "integer"]

        if column not in dtype_dict or dtype_dict[column] not in numeric_types:
            print(f"Column '{column}' is not numeric")
            return self

        col_ref = F.col(column)

        # Build condition depending on provided bounds
        if lower is not None and upper is not None:
            condition = col_ref.between(lower, upper)
        elif lower is not None:
            condition = col_ref >= lower
        else:
            condition = col_ref <= upper

        # Handle NULL values
        condition = F.when(col_ref.isNull(), None).otherwise(condition)

        # Append validation column
        new_col = f"{column}_valid_range"

        self.df = self.df.withColumn(new_col, condition)

        return self
    
    # Method that checks if each value in a string column falls within a user specified set of levels 
    def check_string_levels(self, column, levels):

        # Check column type
        dtype_dict = dict(self.df.dtypes)

        if column not in dtype_dict or dtype_dict[column] != "string":
            print(f"Column '{column}' is not a string column")
            return self

        col_ref = F.col(column)

        # Check string is in allowed levels
        condition = col_ref.isin(levels)

        # Handle NULL values
        condition = F.when(col_ref.isNull(), None).otherwise(condition)

        # Create new column name
        new_col = f"{column}_valid_levels"

        # Append column
        self.df = self.df.withColumn(new_col, condition)

        return self
    
    # Method to check for NA rows & add boolean col
    def check_missing(self, column):

        # Check that column exists
        if column not in self.df.columns:
            print(f"Column '{column}' does not exist")
            return self

        col_ref = F.col(column)

        # Check for NULL values
        condition = col_ref.isNull()

        # Name for appended column
        new_col = f"{column}_is_na"

        # Append column
        self.df = self.df.withColumn(new_col, condition)

        return self
    
    # Min/max stats summary
    def min_max_summary(self, column=None, group=None):

        dtype_dict = dict(self.df.dtypes)
        numeric_types = ["int", "bigint", "double", "float", "long", "integer"]

        # If a specific column is supplied
        if column is not None:
            # Check if num
            if column not in dtype_dict or dtype_dict[column] not in numeric_types:
                print(f"Column '{column}' is not numeric")
                return None
            
            # Check for group
            if group is not None:
                result = (self.df.groupBy(group).agg(F.min(column).alias("min"),
                        F.max(column).alias("max")))
            else:
                result = (self.df.agg(F.min(column).alias("min"),
                        F.max(column).alias("max")))

            return result.toPandas()

        # If no column supplied summarize all numeric columns
        else:
            numeric_cols = [c for c, t in self.df.dtypes if t in numeric_types]

            if group is not None:
                summaries = []

                for col in numeric_cols:
                    temp = (self.df.groupBy(group).agg( F.min(col).alias(f"{col}_min"),
                            F.max(col).alias(f"{col}_max")).toPandas())
                    
                    # append to df
                    summaries.append(temp)

                return reduce(lambda left, right: pd.merge(left, right, on=group), summaries)

            else:
                agg_exprs = []

                for col in numeric_cols:
                    agg_exprs.append(F.min(col).alias(f"{col}_min"))
                    agg_exprs.append(F.max(col).alias(f"{col}_max"))

                result = self.df.agg(*agg_exprs)

                return result.toPandas()
            
    # Return counts
    def count_levels(self, column1, column2=None):

        dtype_dict = dict(self.df.dtypes)

        # Check first column is string
        if column1 not in dtype_dict or dtype_dict[column1] != "string":
            print(f"'{column1}' is not a string")
            return None

        # If second column check it too
        if column2 is not None:
            if column2 not in dtype_dict or dtype_dict[column2] != "string":
                print(f" '{column2}' is not a string")
                return None

            result = (self.df.groupBy(column1, column2).count())

        else:
            result = (self.df.groupBy(column1).count())

        return result.toPandas()
    
    # Return counts - boolean
    def count_levels_bool(self, column1, column2=None):

        dtype_dict = dict(self.df.dtypes)

        # Check first column is string
        if column1 not in dtype_dict or dtype_dict[column1] != "boolean":
            print(f"'{column1}' is not boolean")
            return None

        # If second column check it too
        if column2 is not None:
            if column2 not in dtype_dict or dtype_dict[column2] != "boolean":
                print(f" '{column2}' is not boolean")
                return None

            result = (self.df.groupBy(column1, column2).count())

        else:
            result = (self.df.groupBy(column1).count())

        return result.toPandas()