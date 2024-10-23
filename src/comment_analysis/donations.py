"""src.comment_analysis.donations.py -- module for donation processing."""

import logging
import pandas as pd

from bs4 import BeautifulSoup
from currency_converter import CurrencyConverter
from typing import List

from src.comment_analysis.exceptions import NoDonationsFoundException


logger = logging.getLogger(__name__)


# dictionary for converting symbols to codes - extend if required

symbol_to_code_dictionary = {
    '€': 'EUR',
    '$': 'USD',
    '£': 'GBP',
    '¥': 'JPY',
    '₹': 'INR',
    '₩': 'KRW',
    '₦': 'NGN',
    '₽': 'RUB',
    'R$': 'BRL',
}


class DonationProcessor:
    """Class to process donations from raw HTML."""
    
    def __init__(self,
                 target_currency: str = 'USD'):
        """
        Initialise the class.
        
        :param target_currency: currency to which to convert the donations ('USD' by default).
        """
        self._target_currency = target_currency
    
    def process_donations(self, 
                          soup: BeautifulSoup) -> dict:
        """
        Method to process donations based on parsed HTML data.
        
        :param soup: parsed HTML data.
        
        :return: dictionary with donation DF and donation metrics.
        """
        donation_df = self._get_donations_from_html(soup=soup)

        if len(donation_df) == 0:
            raise NoDonationsFoundException

        unique_currencies = donation_df['currency'].unique()
        
        curr_conv_df = self._get_currency_conversion_df(unique_currencies=unique_currencies)
        
        merged_donation_df = donation_df.merge(curr_conv_df, left_on='currency', right_on='currency', how='left')
        merged_donation_df['conv_donation'] = merged_donation_df['donation'] * merged_donation_df['factor']

        logger.info(f"Total sum of donations: {merged_donation_df['conv_donation'].sum():.2f} {self._target_currency}\n")
        
        return {
            'details': {
                'donation_df': merged_donation_df
            },
            'metrics': {
                'donations__count': len(merged_donation_df),
                'donations__sum': round(merged_donation_df['conv_donation'].sum(), 2),
                'donations__currency': self._target_currency
            }
        }
        
    def _get_donations_from_html(self,
                                 soup: BeautifulSoup) -> pd.DataFrame:
        """
        Isolate donations from bs4 object.

        :param soup: BeautifulSoup object.

        :return: pd.DataFrame with donation data.
        """
        price_spans = soup.findAll('span', {'id': 'comment-chip-price'})
        donations = [elem.text.strip() for elem in price_spans if len(elem.text.strip()) > 0]
        donation_df = pd.DataFrame(
            [elem.split("\xa0") for elem in donations], 
            columns=['donation', 'currency']
        )
        donation_df['donation'] = donation_df['donation'].apply(
            lambda donation: donation.replace(',', '.')
        ).astype(float)

        return donation_df
    
    def _get_currency_conversion_df(self,
                                    unique_currencies: List[str]) -> pd.DataFrame:
        """
        Get a DataFrame with conversion rates for the currencies of the donations on a video.

        :param unique_currencies: list of unique currencies.

        :return: DataFrame with conversion rates.
        """
        currency_rates = CurrencyConverter()

        conv_rates = []
        for currency in unique_currencies:
            try:
                conv_rates.append(
                    {
                        'currency': currency,
                        'factor': currency_rates.convert(
                            amount=1,
                            currency=symbol_to_code_dictionary.get(currency, currency), 
                            new_currency=self._target_currency
                        )
                    }
                )
            except ValueError as e:
                raise ValueError(f"Invalid currency provided: {currency}")

        return pd.DataFrame(conv_rates)    
    