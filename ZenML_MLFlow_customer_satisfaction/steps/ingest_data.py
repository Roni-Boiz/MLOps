import logging

import pandas as pd
from zenml import step


class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self, data_path: str) -> None:
        """
        Args:
            data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        """
        Ingesting data to data_path
        """
        logging.info(f"Ingesting data from {self.data_path}")
        df = pd.read_csv(self.data_path, parse_dates=['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date','shipping_limit_date'], date_format = '%Y-%m-%d %H:%M:%S')
        return df


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Args:
        data_path: path to the data
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e