# Copyright © 2025 Lane Hale. All rights reserved.

### Imports ###
import copy
import isodate
import json
import logging
import os
import pytz
import random
import re
import requests
import torch
import uuid

from flask import Flask, request, jsonify
from datetime import datetime, date, timedelta
from dateutil import parser
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR, SA, SU
from timezonefinder import TimezoneFinder
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


app = Flask(__name__)

# Initialize TimezoneFinder once
timezone_finder = TimezoneFinder()

# Configure the root logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


#############################################
#   Prepare Model Classification Pipeline   #
#############################################

# Define ID to label mapping
id2label_mapping = {0: "booking", 1: "general", 2: "status"}

model_path = "lanehale1/airline_chatbot_model"  # Use HF Models repo

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    id2label=id2label_mapping,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

question_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


#########################
#   Prepare Data Maps   #
#########################

# Copy JSON Files to Dictionaries
file_path = "airline_code_to_name.json"
with open(file_path, "r") as file:
    carrier_codes = json.load(file)

file_path = "airline_name_and_code_to_code.json"
with open(file_path, "r") as file:
    carrier_dual_mapping = json.load(file)

file_path = "airport_code_to_name.json"
with open(file_path, "r") as file:
    airport_codes = json.load(file)

file_path = "airport_name_and_code_to_code.json"
with open(file_path, "r") as file:
    airport_dual_mapping = json.load(file)

file_path = "us_states_dict.json"
with open(file_path, "r") as file:
    us_states_dict = json.load(file)

file_path = "us_state_mappings.json"
with open(file_path, "r") as file:
    us_state_mappings = json.load(file)

# Sort carrier names/codes by length in descending order to prioritize longer matches
sorted_carriers = sorted(
    carrier_dual_mapping.items(), key=lambda item: len(item[0]), reverse=True
)

# Sort airport names/codes by length in descending order to prioritize longer matches
sorted_airport_mappings = sorted(
    airport_dual_mapping.items(), key=lambda item: len(item[0]), reverse=True
)


##############################
#   Compile Regex Patterns   #
##############################

# Build a regex pattern for all unique state names and abbreviations from us_states_dict
state_patterns = []
for abbr, name in us_states_dict.items():
    state_patterns.append(re.escape(name))
    state_patterns.append(re.escape(abbr))
# Sort by length to prioritize longer matches
state_regex = (
    r"\b(?:" + "|".join(sorted(state_patterns, key=len, reverse=True)) + r")\b"
)
STATE_REGEX = re.compile(state_regex, re.IGNORECASE)

COMMA_COLON_REGEX = re.compile(r"[,:]")
DASH_SLASH_REGEX = re.compile(r"[-/]")

# Define a regex for impossible day values 32-99 in MM/DD
impossible_day_regex = r"\b(?:(?:\d{1,2}[-/])|(?:))(3[2-9]|[4-9]\d)(?:,)?\b"
IMPOSSIBLE_DAY_REGEX = re.compile(impossible_day_regex, re.IGNORECASE)

# ---------------------------------------------------------------------------- #
#   Define Date Patterns for extract_date_info and standardize_date_strings    #
# ---------------------------------------------------------------------------- #
# --- Define variables to build larger regex patterns ---
# Month names
month_names = (
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    + r"January|February|March|April|May|June|July|August|September|October|November|December)"
)
# Time intervals for relative dates
intervals = r"(?:days|day|weeks|week|months|month|years|year)"

# Days of the week
weekdays = (
    r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|"
    + r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
)
# Numbers and words for numbers
numbers = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|couple|few|the|a)"

# Map text numbers to integers
text_numbers_map = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "couple": 2,
    "few": 3,
    "the": 1,
    "a": 1,
}

# Map weekdays to relativedelta codes
weekday_map = {
    "monday": MO,
    "tuesday": TU,
    "wednesday": WE,
    "thursday": TH,
    "friday": FR,
    "saturday": SA,
    "sunday": SU,
    "mon": MO,
    "tue": TU,
    "wed": WE,
    "thu": TH,
    "fri": FR,
    "sat": SA,
    "sun": SU,
}

# --- Define relative date patterns ---
# 3 days from now, two weeks from Mon, a week after tomorrow
relative_interval_from_pattern = (
    r"(?:in\s+)?"
    + numbers
    + r"\s+"
    + intervals
    + r"\s+"
    + r"(?:from|after)\s+(?:now|today|tomorrow|"
    + weekdays
    + r")"
)
# next Friday, next Sat.
relative_next_weekday_pattern = r"next\s+" + weekdays

# in 3 days, in two weeks, in a month
relative_in_interval_pattern = r"in\s+" + numbers + r"\s+" + intervals

# tomorrow, various ways of saying 'today'
relative_simple_terms = (
    r"(?:tomorrow|today|tonight|now|this\safternoon|this\sevening|"
    + r"this\smorning|immediately|asap|this\sminute|this\sinstant|"
    + r"as\ssoon\sas\spossible|at\sonce|straight\saway|presently)"
)
RELATIVE_INTERVAL_FROM_PATTERN = re.compile(
    relative_interval_from_pattern, re.IGNORECASE
)
RELATIVE_NEXT_WEEKDAY_PATTERN = re.compile(relative_next_weekday_pattern, re.IGNORECASE)
RELATIVE_IN_INTERVAL_PATTERN = re.compile(relative_in_interval_pattern, re.IGNORECASE)
RELATIVE_SIMPLE_TERMS = re.compile(relative_simple_terms, re.IGNORECASE)

ORDINAL_SUFFIX_REGEX = re.compile(r"(?<=\d)(st|nd|rd|th)\b")
OF_REGEX = re.compile(r"\bof\b")

# --- Define all the regex patterns with descriptive names for common date formats ---
# YYYY/MM/DD or YYYY-MM-DD
numeric_ymd = r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
# MM/DD/YY or MM-DD-YYYY
numeric_mdy = r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"
# MM/DD or MM-DD
numeric_md = r"\b\d{1,2}[-/]\d{1,2}\b"
# The next five patterns allow optional dot after Month, and optional comma before optional Year
# January 8, January 8, 2023, jan. 8
month_day_year = r"\b" + month_names + r"\.?\s+\d{1,2}(?:,?\s+(?P<year>\d{4}))?\b"
# 8 January, 8 January 2023, 8jan
day_month_year = r"\b\d{1,2}\s*" + month_names + r"(?:\.?,?\s+(?P<year>\d{4}))?\b"
# 1st January, 22nd February 2023
day_ordinal_month_year = (
    r"\b\d{1,2}(?:st|nd|rd|th)\s*" + month_names + r"(?:\.?,?\s+(?P<year>\d{4}))?\b"
)
# June 1st, December 22nd, 2023
month_day_ordinal_year = (
    r"\b" + month_names + r"\.?\s+\d{1,2}(?:st|nd|rd|th)(?:,?\s+(?P<year>\d{4}))?\b"
)
# 1st of June, 22nd of December 2023
day_ordinal_of_month_year = (
    r"\b\d{1,2}(?:st|nd|rd|th)\s+of\s+"
    + month_names
    + r"(?:\.?,?\s+(?P<year>\d{4}))?\b"
)
# 3 days from now, next Friday, in two weeks, tomorrow
relative_date = (
    r"\b(?:"  # Start non-capturing group for alternatives
    + relative_interval_from_pattern
    + r"|"
    + relative_next_weekday_pattern
    + r"|"
    + relative_in_interval_pattern
    + r"|"
    + relative_simple_terms
    + r")\b"  # End non-capturing group and word boundary
)

NUMBERIC_YMD = re.compile(numeric_ymd, re.IGNORECASE)
NUMBERIC_MDY = re.compile(numeric_mdy, re.IGNORECASE)
NUMBERIC_MD = re.compile(numeric_md, re.IGNORECASE)
MONTH_DAY_YEAR = re.compile(month_day_year, re.IGNORECASE)
DAY_MONTH_YEAR = re.compile(day_month_year, re.IGNORECASE)
DAY_ORDINAL_MONTH_YEAR = re.compile(day_ordinal_month_year, re.IGNORECASE)
MONTH_DAY_ORDINAL_YEAR = re.compile(month_day_ordinal_year, re.IGNORECASE)
DAY_ORDINAL_OF_MONTH_YEAR = re.compile(day_ordinal_of_month_year, re.IGNORECASE)
RELATIVE_DATE = re.compile(relative_date, re.IGNORECASE)

# Define a dictionary of regex patterns with descriptive names
date_patterns = {
    "numeric_ymd": NUMBERIC_YMD,
    "numeric_mdy": NUMBERIC_MDY,
    "numeric_md": NUMBERIC_MD,
    "month_day_year": MONTH_DAY_YEAR,
    "day_month_year": DAY_MONTH_YEAR,
    "day_ordinal_month_year": DAY_ORDINAL_MONTH_YEAR,
    "month_day_ordinal_year": MONTH_DAY_ORDINAL_YEAR,
    "day_ordinal_of_month_year": DAY_ORDINAL_OF_MONTH_YEAR,
    "relative_date": RELATIVE_DATE,
}
# --- End of Patterns for extract_date_info and standardize_date_strings ---

# 2-char carrier code, no space, 1 to 4 digits flight number (e.g., AA123)
CC9999_REGEX = re.compile(r"\b([A-Z]{2})(\d{1,4})\b", re.IGNORECASE)
# 1 to 4 digits
FLIGHT_NUMBER_REGEX = re.compile(r"\d{1,4}")

# Create a dictionary of common word variations to replace
word_replacements = {
    r"D\.C\.": "Washington",
    r"K\.C\.": "Kansas City",
    r"L\.A\.": "Los Angeles",
    r"N\.Y\.C\.": "New York",
    r"\bSaint\b": "St.",
    r"\bSt\s": "St. ",
    r"\bFt\.\s*": "Fort ",
    r"\bFt\s": "Fort ",
    # Add other common variations as needed (e.g., 'Mount' to 'Mt.')
}
# Sort replacement patterns by length in descending order
sorted_replacements = sorted(
    word_replacements.items(), key=lambda item: len(item[0]), reverse=True
)
# Compile the patterns and create a new list of (compiled_pattern, replacement) tuples
COMPILED_REPLACEMENTS = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in sorted_replacements
]

# 3 capital letters (e.g, LAX)
AIRPORT_CODE_REGEX = re.compile(r"[A-Z]{3}")
# 3 letters or a number 0 to 99
OVER_9_AIRPORTS_REGEX = re.compile(r"\b(?:[A-Z]{3}|[0-9]{1,2})\b", re.IGNORECASE)
# '1 stop' or '2 stops'
NUMBER_OF_STOPS_REGEX = re.compile(r"\(.*,\s*(\d+)\s*stop")


#################################################################################
###                               Functions                                   ###
#################################################################################
#   choose_retry_phrase()                                                       #
#   remove_duplicate_locations_by_city_name(found_locations)                    #
#   remove_duplicate_locations_by_airport_code(found_locations)                 #
#   find_matching_state(search_string, location)                                #
#   check_same_name_cities(found_locations, temp_query, errors)                 #
#   build_airport_list(airport_info, airport_code, airport_list, found_states)  #
#   parse_query_date(date_string)                                               #
#   extract_date_info(query)                                                    #
#   standardize_date_strings(date_dict, state_dict)                             #
#   extract_flight_info(query, intent, state_dict)                              #
#   get_flight_status(flight_ident)                                             #
#   get_flights_to_book(flight_info, passengers)                                #
#   get_timezone_name(latitude, longitude, tz_finder=timezone_finder)           #
#   get_local_datetime(datetime_string, local_timezone_str)                     #
#   append_departure_details(flight, response_lines)                            #
#   append_arrival_details(flight, response_lines)                              #
#   format_delay_time(delay_seconds)                                            #
#   build_status_response(flight_data, flight_ident)                            #
#   build_booking_response(booking_data, departure_date, tz_finder)             #
#   update_relative_dates(state_dict)                                           #
#   parse_multi_airport_response(user_input, state_dict)                        #
#   sort_flight_options(flight_options, sort_radio)                             #
#   build_flight_options_batch(sorted_options, start_index, batch_size=10)      #
#   show_more_flight_options(chat_history, state_dict)                          #
#   chat_flight_assistant(chat_history, state_dict, user_input)                 #
#################################################################################


######################################
#   Choose Retry Phrase variations   #
######################################
def choose_retry_phrase():
    """
    Chooses a random retry phrase from a list of retry phrases.
    Returns:
        str: A random retry phrase.
    """
    retry_phrases = [
        "Please try again.",
        "Could you please retry?",
        "Could you try again?",
        "Please check and give it another try.",
        "Please try your request again.",
        "Could you please try again?",
        "Please recheck and try again.",
    ]
    return random.choice(retry_phrases)


###############################################
#   Remove Duplicate Locations by City Name   #
###############################################
def remove_duplicate_locations_by_city_name(found_locations):
    """
    Removes duplicate found locations based on city names.
    Args:
        found_locations (list): A list of found locations.
    Returns:
        unique_locations (list): A list of unique found locations.
    """
    # Check for user inputs like "Chicago ORD", "Chicago-ORD", "Chicago, ORD",
    # "Chicago (ORD)", "Chicago / ORD", or the reverse, "ORD Chicago", etc.
    unique_locations = []
    i = 0
    while i < len(found_locations):
        current_location = found_locations[i]

        if (
            i + 1 < len(found_locations)
            and current_location["city"] == found_locations[i + 1]["city"]
            and (
                (
                    current_location["input_was_airport_code"]
                    and not found_locations[i + 1]["input_was_airport_code"]
                )
                or (
                    not current_location["input_was_airport_code"]
                    and found_locations[i + 1]["input_was_airport_code"]
                )
            )
            and current_location["end_index"] + 3
            >= found_locations[i + 1]["start_index"]
        ):
            # Found a city name and airport code for the same city close together
            if current_location["input_was_airport_code"]:
                # Keep the one that was an airport code
                unique_locations.append(current_location)
            else:
                # The next one must be the airport code
                unique_locations.append(found_locations[i + 1])
            i += 2  # Skip both the city name and airport code
        else:
            # No duplicate city/airport code found, keep the current location
            unique_locations.append(current_location)
            i += 1

    return unique_locations


##################################################
#   Remove Duplicate Locations by Airport Code   #
##################################################
def remove_duplicate_locations_by_airport_code(found_locations):
    """
    Removes duplicate found locations based on airport codes.
    Args:
        found_locations (list): A list of found locations.
    Returns:
        unique_locations (list): A list of unique found locations.
    """
    # This should catch inputs like "Springfield, IL SPI", "SPI Springfield, IL-SPI",
    # or "SPI - Springfield-SPI to SGF" after the state has been checked
    unique_locations = []
    i = 0
    while i < len(found_locations):
        current_location = found_locations[i]
        if (
            i + 1 < len(found_locations)
            and len(current_location["airport"]) == 1
            and len(found_locations[i + 1]["airport"]) == 1
            and current_location["airport"][0]["code"]
            == found_locations[i + 1]["airport"][0]["code"]
        ):
            # Found a city name and airport code for the same city close together
            unique_locations.append(found_locations[i + 1])
            i += 2
        else:
            # No duplicate city/airport code found, keep the current location
            unique_locations.append(current_location)
            i += 1

    return unique_locations


###########################
#   Find Matching State   #
###########################
def find_matching_state(search_string, location):
    """
    Finds a matching state in a search string and updates the location dictionary.
    Args:
        search_string (str): The string to search for a matching state.
        location (dict): The dictionary containing location information.
    Returns:
        found_flag (bool): True if a matching state is found and updated, False otherwise.
    """
    found_flag = False
    match = STATE_REGEX.search(search_string)

    if match:
        state_match = match.group(0).upper()
        end_index = match.end()

        if state_match in us_state_mappings:
            state_abbr = us_state_mappings[state_match]
            city_state = location["city"] + ", " + state_abbr
            airports = []

            for airport in location["airport"]:
                if airport["state"] == state_abbr:
                    airports.append(airport)

            if airports:
                location["airport"] = airports
                location["city"] = city_state
                location["end_index"] = end_index
                found_flag = True

    return found_flag


##############################
#   Check Same Name Cities   #
##############################
def check_same_name_cities(found_locations, temp_query, errors):
    """
    Looks for a state in the input for cities with the same name in different states.
    Args:
        found_locations (list): A list containing information about found locations.
        temp_query (str): The user's query without dates.
        errors (list): A list to store error messages.
    Modifies:
        found_locations (list): Updates city to city, state. Removes unrelated airports.
        errors (list): Appends error messages if no state is found.
    """
    errors_dict = {}
    for i in range(len(found_locations)):
        location = found_locations[i]
        num_airports = len(location["airport"])
        if num_airports > 1 and not location["same_state_multi_airport_city"]:
            window_start = location["end_index"]
            if i < len(found_locations) - 1:
                window_end = found_locations[i + 1]["start_index"]
            else:
                window_end = len(temp_query)
            search_string = temp_query[window_start:window_end]
            if not find_matching_state(search_string, location):
                errors_dict[location["city"]] = location["airport"]

    first_message = False
    for city, airports in errors_dict.items():
        state = airports[0]["state"]
        code = airports[0]["code"]
        if not first_message:
            first_message = True
            errors.append(
                f"- Multiple locations have a city named '{city}'. To help me find the "
                + f"correct one, please add a state or airport code from the list "
                + f"below to your entry (e.g., '{city}, {state}' or simply '{code}'):"
            )
        else:
            errors.append(
                f"- Multiple locations have a city named '{city}'. Please include a "
                + f"state or airport code from the following:"
            )
        for airport in airports:
            city_airport_str = airport_codes[airport["code"]] + f" ({airport['code']})"
            # Nest these bullet points by indenting 4 spaces
            errors.append(f"    - {city_airport_str}")

    return None


##########################
#   Build Airport List   #
##########################
def build_airport_list(airport_info, airport_code, airport_list, found_states):
    """
    Updates the airport list associated with a given city name.
    Args:
        airport_info (str): The airport information (from airport_codes dict).
        airport_code (str): The airport code.
        airport_list (list): The list of airports.
        found_states (set): The set of found states.
    Returns:
        city_names[0] (str): The first city name associated with the airport.
    Modifies:
        airport_list (list): Appends the airport details.
        found_states (set): Adds the airport's state.
    """
    # Split by comma and colon for "City, State: Airport Name"
    airport_info_parts = [
        part.strip() for part in COMMA_COLON_REGEX.split(airport_info)
    ]
    airport_name = ""
    city_names = airport_info_parts[0].split("/")
    airport_state = airport_info_parts[1]
    airport_name = airport_info_parts[2]
    airport_list.append(
        {
            "code": airport_code,
            "name": airport_name,
            "state": airport_state,
        }
    )
    found_states.add(airport_state)
    return city_names[0]


########################
#   Parse Query Date   #
########################
def parse_query_date(date_string, includes_year=False):
    """
    Parses a date string into a datetime object using dateutil.
    If the date string does not include a year, it defaults to the current year.
    If the resulting date is before today's date (excluding time),
    it increments the year to handle future dates entered without a year.
    Includes validation to ensure the parsed date is a valid calendar date.
    Args:
        date_string (str): The date string to parse.
    Returns:
        parsed_datetime (datetime): A datetime object representing the parsed date,
            or None if parsing fails or the date is invalid.
    """
    try:
        # --- Initial Validation ---
        # Check for impossible day values 32-99 in MM/DD (extract_date_info
        # doesn't allow days over two digits or 2-digit years with month names).

        if IMPOSSIBLE_DAY_REGEX.search(date_string):
            logging.error(
                f"parse_query_date: Found impossible day value in string: {date_string}"
            )
            return None
        # --- End Initial Validation ---

        # Get today's date without time for comparison
        today_date = date.today()

        # Parse the date string. By default, dateutil will use the current year if none is provided.
        parsed_datetime = parser.parse(date_string)

        # If the parsed date is before today's date, increment the year.
        # We compare only the date part to ignore the time component.
        if not includes_year and parsed_datetime.date() < today_date:
            parsed_datetime = parsed_datetime.replace(year=parsed_datetime.year + 1)

        return parsed_datetime

    except (ValueError, TypeError) as e:
        # Handle cases where parsing might fail
        logging.error(
            f"parse_query_date: Could not parse date string: {date_string}. Error: {e}"
        )
        return None


####################################
#   Extract Date Info from query   #
####################################
def extract_date_info(query):
    """
    Extracts dates in various formats from a natural language query using
    a list of regex patterns, indicating which pattern matched each date.
    Args:
        query (str): The natural language query.
    Returns:
        date_dict (dict): A dictionary containing the extracted date information,
             including the date string, the matched pattern type, and start index.
    """
    date_dict = {}
    found_dates = []
    temp_query = query

    # Iterate through the compiled regex patterns and find all matches
    for pattern_name, COMPILED_PATTERN in date_patterns.items():
        # Use re.finditer to find all non-overlapping matches
        for match in COMPILED_PATTERN.finditer(temp_query):
            start_index = match.start()
            end_index = match.end()
            date_str = match.group(0)

            if "year" in match.groupdict() and match.group("year") is not None:
                includes_year = True
            elif pattern_name in ["numeric_mdy", "numeric_ymd"]:
                includes_year = True
            else:
                includes_year = False

            # Add the found date information to the list
            found_dates.append(
                {
                    "date": date_str,
                    "includes_year": includes_year,
                    "pattern_type": pattern_name,
                    "start_index": start_index,
                    "end_index": end_index,
                }
            )

            # Replace the matched substring with unmatchable text
            replacement = "~" * len(date_str)
            temp_query = temp_query[:start_index] + replacement + temp_query[end_index:]

    # Sort the found dates by their appearance in the original query
    found_dates.sort(key=lambda x: x["start_index"])

    date_dict["extracted_dates"] = found_dates
    date_dict["query_without_date"] = temp_query

    return date_dict


############################
#   Standardize Dates   #
############################
def standardize_date_strings(date_dict, state_dict):
    """
    Standardizes extracted date strings into datetime objects.
    Interprets relative date phrases and parses standard date formats.
    Args:
        date_dict (dict): The dictionary returned by extract_date_info.
        state_dict (dict): A dictionary containing state-related information.
    Returns:
        standardized_dates (list): A list of datetime objects representing the
            standardized dates.
        invalid_dates (list): A list of invalid date strings if standardization fails.
    """
    standardized_dates = []
    invalid_dates = []

    today = state_dict["user_date"] or date.today()

    for date_info in date_dict["extracted_dates"]:
        date_str = date_info["date"].lower()
        pattern_type = date_info["pattern_type"]
        includes_year = date_info["includes_year"]

        if pattern_type == "relative_date":
            # Handle relative date phrases - check for longer phrases first
            match_interval_from = RELATIVE_INTERVAL_FROM_PATTERN.search(date_str)
            match_next_weekday = RELATIVE_NEXT_WEEKDAY_PATTERN.search(date_str)
            match_in_interval = RELATIVE_IN_INTERVAL_PATTERN.search(date_str)
            match_simple_term = RELATIVE_SIMPLE_TERMS.search(date_str)

            if match_interval_from:
                # Use the interval from match
                parts = match_interval_from.group(0).split()
                if parts[0].lower() == "in":
                    del parts[0]  # Remove optional 'in'
                number_str = parts[0]  # Extract number part
                interval = parts[1].rstrip(
                    "s"
                )  # Extract singular interval (day, week, etc.)
                from_when = parts[-1]  # Extract 'now', 'today', or weekday

                if number_str.lower() in text_numbers_map:
                    num = text_numbers_map[number_str.lower()]
                else:
                    num = int(number_str)  # Convert number to integer

                # Calculate the base date to add timedelta/relativedelta to
                if from_when.lower() == "tomorrow":
                    base_date = today + timedelta(days=1)
                elif from_when.lower() == "now" or from_when.lower() == "today":
                    base_date = today
                else:
                    # Find the next occurrence of the specified weekday from today.
                    # relativedelta(weekday=...) finds the next occurrence of that weekday
                    # except if weekday == today's weekday, then it returns today's date.
                    base_date = today + relativedelta(
                        weekday=weekday_map[from_when.lower()]
                    )

                # Save the standardized date
                if interval == "day":
                    standardized_dates.append(base_date + timedelta(days=num))
                elif interval == "week":
                    standardized_dates.append(base_date + timedelta(weeks=num))
                elif interval == "month":
                    standardized_dates.append(base_date + relativedelta(months=num))
                elif interval == "year":
                    standardized_dates.append(base_date + relativedelta(years=num))

            elif match_next_weekday:
                # Use the next weekday match
                parts = match_next_weekday.group(0).split()
                weekday_str = parts[1]
                # Find the next occurrence of the specified weekday from today.
                # relativedelta(weekday=...) finds the next occurrence of that weekday
                # except if weekday == today's weekday, then it returns today's date.
                next_weekday_date = today + relativedelta(
                    weekday=weekday_map[weekday_str.lower()]
                )
                if next_weekday_date == today:
                    standardized_dates.append(next_weekday_date + timedelta(weeks=1))
                else:
                    standardized_dates.append(next_weekday_date)

            elif match_in_interval:
                # Use the in interval match
                parts = match_in_interval.group(0).split()
                number_str = parts[1]  # Extract number part after "in"
                interval = parts[2].rstrip(
                    "s"
                )  # Extract singular interval (day, week, etc.)

                if number_str.lower() in text_numbers_map:
                    num = text_numbers_map[number_str.lower()]
                else:
                    num = int(number_str)  # Convert number to integer

                # Save the standardized date
                if interval == "day":
                    standardized_dates.append(today + timedelta(days=num))
                elif interval == "week":
                    standardized_dates.append(today + timedelta(weeks=num))
                elif interval == "month":
                    standardized_dates.append(today + relativedelta(months=num))
                elif interval == "year":
                    standardized_dates.append(today + relativedelta(years=num))

            elif match_simple_term:
                # Use the simple term match
                if match_simple_term.group(0).lower() == "tomorrow":
                    standardized_dates.append(today + timedelta(days=1))
                else:
                    standardized_dates.append(today)

            else:
                logging.warning(
                    f"standardize_date_strings: Unhandled relative date phrase: {date_str}"
                )

        else:
            # Preprocess ordinal dates and "of" before parsing standard formats
            processed_date_str = date_str

            if pattern_type in [
                "day_ordinal_month_year",
                "month_day_ordinal_year",
                "day_ordinal_of_month_year",
            ]:
                # Remove ordinal suffixes (st, nd, rd, th)
                processed_date_str = ORDINAL_SUFFIX_REGEX.sub("", processed_date_str)
                # Remove "of" if present
                processed_date_str = OF_REGEX.sub("", processed_date_str).strip()
                # Parse the processed date string
                parsed_date = parse_query_date(processed_date_str, includes_year)

            elif pattern_type in [
                "numeric_ymd",
                "numeric_mdy",
                "numeric_md",
            ]:
                # Swap DD with MM if international date format
                if state_dict["preferred_date_format"] == "DD/MM":
                    split_date = DASH_SLASH_REGEX.split(processed_date_str)
                    if pattern_type == "numeric_md":  # Change dm to md
                        processed_date_str = f"{split_date[1]}/{split_date[0]}"
                    elif pattern_type == "numeric_mdy":
                        # This pattern could match MM/DD/YY or YY/MM/DD but we can't determine
                        # which without asking the user. YY/MM/DD is quite rare, even in parts
                        # of Asia. The standard format in East Asian countries like China, Japan,
                        # and Korea is the four-digit year, or YYYY/MM/DD, which aligns with the
                        # ISO 8601 international standard. Year/Day/Month is not standard in any
                        # region and would be extremely rare to encounter.

                        # We'll ignore possible YY/MM/DD (and YY/DD/MM) inputs. However, some will
                        # work: e.g., 26/3/2 is matched as numeric_md since '2' doesn't satisfy the
                        # 2 or 4-digit year requirement of numeric_mdy. 26/3 becomes 3/26 above or
                        # or March 26, 2026 (today being 10/24/2025).

                        processed_date_str = (  # Change dmy to mdy
                            f"{split_date[1]}/{split_date[0]}/{split_date[2]}"
                        )
                    # Else the pattern_type is numeric_ydm. Year/Day/Month is not standard in any
                    # region and would be extremely rare to encounter. Even DD/MM users would use
                    # YYYY/MM/DD if putting the year first. We'll treat YYYY/DD/MM as invalid.

                parsed_date = parse_query_date(processed_date_str, includes_year)

            else:
                # Parse the unprocessed date string
                parsed_date = parse_query_date(processed_date_str, includes_year)

            # Save just the datetime.date part if valid
            if parsed_date:
                standardized_dates.append(parsed_date.date())
            else:
                invalid_dates.append(date_str)

    return standardized_dates, invalid_dates


######################################
#   Extract Flight Info from query   #
######################################
def extract_flight_info(query, intent, state_dict):
    """
    Extracts flight information from a natural language query.
    Args:
        query (str): The natural language query.
        intent (str): The intent of the query ('status' or 'booking').
        state_dict (dict): A dictionary containing state-related information.
    Returns:
        flight_info (dict): A dictionary containing the extracted flight information.
        assistant_response (str): A string containing error messages.
    """
    assistant_response = ""
    errors = []

    # ---------------------------------------#
    #   General: return a helpful response   #
    # ---------------------------------------#
    if intent == "general":

        assistant_response = (
            "I'm sorry, I didn't understand. I can help with flight status or finding "
            + "flights, but I'm unable to handle other types of requests for now."
        )

        flight_info = {}

    # --------------------------------------------------------------------------#
    #   Status: Extract airline code (e.g., AA) and flight number (e.g., 123)   #
    # --------------------------------------------------------------------------#
    elif intent == "status":

        flight_info = {"airline_code": None, "flight_number": None, "flight_date": None}

        # Look for date info and remove it from the query
        date_dict = extract_date_info(query)

        if date_dict["extracted_dates"]:
            query = date_dict["query_without_date"]
            valid_dates, invalid_dates = standardize_date_strings(date_dict, state_dict)
            if valid_dates:
                flight_info["dates"] = valid_dates[0].strftime("%Y-%m-%d")

        # Remove hyphens and slashes from the query
        query_split = DASH_SLASH_REGEX.split(query)
        query = " ".join(query_split)

        for name_or_code, codes in sorted_carriers:
            # Use regex to find whole word matches for names and codes
            word_pattern = r"\b" + re.escape(name_or_code) + r"\b"
            compiled_word_pattern = re.compile(word_pattern, re.IGNORECASE)
            match = compiled_word_pattern.search(query)
            if match:
                flight_info["airline_code"] = codes
                break  # Once an airline is found, we can stop looking for others

        if flight_info["airline_code"] is None:
            # Fallback to matching AA123 format if no specific name/code match found
            match = CC9999_REGEX.search(query)
            if match:
                # Ensure airline code is uppercase
                flight_info["airline_code"] = match.group(1).upper()
                flight_info["flight_number"] = match.group(2)

        if flight_info["flight_number"] is None:
            # If an airline code was found by name/code, now look for the flight number
            match = FLIGHT_NUMBER_REGEX.search(query)
            if match:
                flight_info["flight_number"] = match.group(0)

        # Check for errors
        if flight_info["airline_code"] is None:
            if flight_info["flight_number"] is None:
                assistant_response = (
                    "Sorry, I couldn't find a valid airline name/code "
                    + "or flight number in your status request."
                )
                assistant_response += f"\n\n{choose_retry_phrase()}"
            else:
                assistant_response = (
                    "Sorry, I couldn't find a valid airline name/code "
                    + "in your status request."
                )
                assistant_response += f"\n\n{choose_retry_phrase()}"
        elif flight_info["flight_number"] is None:
            assistant_response = (
                "Sorry, I couldn't find a flight number in your status request."
            )
            assistant_response += f"\n\n{choose_retry_phrase()}"

    # --------------------------------------------------------------------------------#
    #   Booking: Extract city or airport codes (e.g., Denver to Chicago → DEN, ORD)   #
    # --------------------------------------------------------------------------------#
    elif intent == "booking":

        # Normalize common variations in city/place names
        normalized_query = query
        # Apply replacements to the query
        for COMPILED_PATTERN, replacement in COMPILED_REPLACEMENTS:
            normalized_query = COMPILED_PATTERN.sub(replacement, normalized_query)

        # Use the normalized_query for further processing
        query = normalized_query

        # Extract date info and remove date(s) from the query
        date_dict = extract_date_info(query)
        query_without_dates = date_dict["query_without_date"]

        # -------------------------------------------------------#
        #   Process location mappings to build found_locations   #
        # -------------------------------------------------------#
        found_locations = []
        for name_or_code, codes in sorted_airport_mappings:

            # Use regex to find whole word matches for names and codes
            word_pattern = r"\b" + re.escape(name_or_code) + r"\b"
            compiled_word_pattern = re.compile(word_pattern, re.IGNORECASE)
            match = compiled_word_pattern.search(query_without_dates)

            # Use a while loop with re.search to find all non-overlapping
            # matches in the continually updated query_without_dates
            while match:
                start_index = match.start()
                end_index = match.end()
                found_states = set()
                airport_list = []

                if codes[0] == "alias":
                    # Switch to city name or airport code key/value
                    name_or_code = codes[1]  # This is the new key
                    codes = airport_dual_mapping[name_or_code]  # New values

                # Determine if the match is an airport code (3 capital letters)
                input_was_airport_code = bool(
                    AIRPORT_CODE_REGEX.fullmatch(name_or_code)
                )

                if input_was_airport_code:
                    # Add airport info for the matched airport code
                    airport_info = airport_codes[name_or_code]
                    city_name = build_airport_list(
                        airport_info, name_or_code, airport_list, found_states
                    )
                    same_state_multi_airport_city = False
                else:
                    # Add all airport codes associated with the matched city name
                    city_name = name_or_code
                    for airport_code in codes:
                        airport_info = airport_codes[airport_code]
                        _ = build_airport_list(
                            airport_info, airport_code, airport_list, found_states
                        )

                    # Determine if it's a multi-airport city within a single state
                    same_state_multi_airport_city = (
                        len(found_states) == 1 and len(airport_list) > 1
                    )

                # Create a dictionary for the found location
                found_locations.append(
                    {
                        "city": city_name,
                        "airport": airport_list,
                        "input_was_airport_code": input_was_airport_code,
                        "same_state_multi_airport_city": same_state_multi_airport_city,
                        "use_all_airports": False,
                        "start_index": start_index,
                        "end_index": end_index,
                    }
                )
                # Replace the matched substring with unmatchable text
                replacement = "*" * len(name_or_code)
                query_without_dates = (
                    query_without_dates[:start_index]
                    + replacement
                    + query_without_dates[end_index:]
                )
                # Search again in the modified string
                match = compiled_word_pattern.search(query_without_dates)

        # Sort found locations by their index in the query to determine origin and destination
        found_locations.sort(key=lambda x: x["start_index"])

        # Check if enough locations were found to form at least one leg
        if not found_locations:
            errors.append(
                "- I didn't find any valid city names or airport codes in your "
                + "entry to form an origin-destination pair.",
            )
        else:
            # Remove duplicate locations
            found_locations = remove_duplicate_locations_by_city_name(found_locations)

            # Check the input for a state after cities with the same name in different states.
            # found_locations is updated if found, or errors is updated if not.
            check_same_name_cities(found_locations, query_without_dates, errors)

            # Check duplicate locations again after state checks
            found_locations = remove_duplicate_locations_by_airport_code(
                found_locations
            )

            if len(found_locations) == 1:
                if found_locations[0]["input_was_airport_code"]:
                    code = found_locations[0]["airport"][0]["code"]
                    errors.append(
                        f"- I only found one valid location ['{code}'], so I "
                        + "couldn't form an origin-destination pair.",
                    )
                else:
                    name = found_locations[0]["city"]
                    errors.append(
                        f"- I only found one valid location ['{name}'], so I "
                        + "couldn't form an origin-destination pair.",
                    )

        # Do date checks after city/airport checks to group errors together
        valid_dates = []
        if date_dict["extracted_dates"]:
            valid_dates, invalid_dates = standardize_date_strings(date_dict, state_dict)

            if not invalid_dates:
                # All dates were successfully parsed
                date_dict["dates"] = valid_dates

            elif len(invalid_dates) == 1:
                # A requested date couldn't be converted to a datetime object
                errors.append(f"- The requested date '{invalid_dates[0]}' is invalid.")
            else:
                # Multiple invalid dates were found
                quoted_invalid_dates = [
                    f"'{date_str}'" for date_str in invalid_dates[:-1]
                ]
                date_string = ", ".join(quoted_invalid_dates)
                date_string += " and " + f"'{invalid_dates[-1]}'"
                errors.append(f"- The requested dates {date_string} are invalid.")
        else:
            # No date was found
            errors.append("- I didn't find a valid departure date in your entry.")
            errors.append(
                "- Try a format like 6/23, 6-23, or Jun 23 (year is optional for any format)."
            )

        # Check for past dates and dates too far in the future
        if valid_dates:
            # Get today's date without time for comparison
            today_date = date.today()
            max_booking_date = today_date + timedelta(days=329)

            for i, date_obj in enumerate(valid_dates):
                # Ignore days too close to today for timezones a day behind UTC
                if date_obj + timedelta(days=2) < today_date:
                    date_str = date_dict["extracted_dates"][i]["date"]
                    errors.append(f"- The requested date '{date_str}' is in the past.")
                elif date_obj > max_booking_date:
                    date_str = date_dict["extracted_dates"][i]["date"]
                    errors.append(
                        f"- The requested date '{date_str}' "
                        + f"({date_obj.strftime('%m-%d-%Y')}) is too far in the future."
                    )
                    errors.append(
                        f"- The furthest date I can search is 329 days from today, "
                        + f"which is currently {max_booking_date.strftime('%m-%d-%Y')}."
                    )
                    break

            # Check dates for chronological order
            if not errors and len(valid_dates) > 1:
                for i in range(len(valid_dates) - 1):
                    # Compare date part only to ignore time component
                    if valid_dates[i] > valid_dates[i + 1]:
                        errors.append(
                            "- The requested dates are not in chronological order."
                        )
                        break

        # Build legs by grouping locations into origin/destination pairs
        if found_locations:
            legs = []
            current_leg = {}
            # Assign locations to origin and destination based on order in query
            for location in found_locations:
                if not current_leg:
                    current_leg["origin"] = location
                else:
                    current_leg["destination"] = location
                    legs.append(copy.deepcopy(current_leg))
                    current_leg = {}
                    current_leg = {"origin": location}

            if legs and "destination" not in legs[-1]:
                del legs[-1]

            # Check for round trip and enough departure dates
            if legs:
                num_legs = len(legs)
                if not "dates" in date_dict:
                    num_dates = 0
                else:
                    num_dates = len(date_dict["dates"])
                    # Check for implied round trip
                    if num_legs == 1 and num_dates == 2:
                        return_leg = {
                            "origin": legs[0]["destination"],
                            "destination": legs[0]["origin"],
                        }
                        legs.append(return_leg)
                        num_legs += 1
                        is_round_trip = True
                    # Check for specified round trip
                    elif num_legs == 2 and legs[0]["origin"] == legs[1]["destination"]:
                        is_round_trip = True
                    else:
                        is_round_trip = False

                # Check for 6 legs max
                if num_legs > 6:
                    errors.append(
                        f"- I found {num_legs} origin-destination pairs in your "
                        + "entry, but I can only search for 6 at a time.",
                    )
                # Add departure dates to each leg
                elif num_dates >= num_legs:
                    for leg, date_obj in zip(legs, date_dict["dates"]):
                        leg["date"] = date_obj
                else:
                    if num_dates == 1:
                        errors.append(
                            f"- I found {num_legs} origin-destination pairs "
                            + f"but only 1 departure date.",
                        )
                    elif num_dates != 0:
                        errors.append(
                            f"- I found {num_legs} origin-destination pairs "
                            + f"but only {num_dates} departure dates.",
                        )

        # Move errors to assistant_response
        if errors:
            if len(errors) == 2 and "Try a format like" in errors[1]:
                special_error = True
            else:
                special_error = False

            if len(errors) == 1 or special_error:
                if errors[0][2] == "I":
                    assistant_response = f"Sorry, {errors[0][2:]}"
                else:
                    first_letter = errors[0][2].lower()
                    assistant_response = f"Sorry, {first_letter}{errors[0][3:]}"

                if special_error:
                    assistant_response += f"\n\n{errors[1][2:]}"
                else:
                    assistant_response += f"\n\n{choose_retry_phrase()}"
            else:
                # More than one error
                errors.insert(0, "Sorry, I couldn't understand your booking request:")
                errors.append("")
                errors.append(choose_retry_phrase())
                assistant_response = "\n".join(errors)

            flight_info = {}

        # Build flight_info
        if not errors:
            # Expedia cabin options are economy/first/business/premiumeconomy
            cabin = "economy"  # Default cabin
            query_lower = query.lower()

            premium_economy_phrase = [
                "premium economy",
                "premium econ",
                "prem econ",
                "prem economy",
            ]
            business_phrase = ["business", "bus.", "biz"]
            first_phrase = ["first class", "1st class", "in first", "in 1st"]

            if any(phrase in query_lower for phrase in premium_economy_phrase):
                cabin = "premiumeconomy"
            elif any(phrase in query_lower for phrase in business_phrase):
                cabin = "business"
            elif any(phrase in query_lower for phrase in first_phrase):
                cabin = "first"

            flight_info = {
                "legs": legs,
                "cabin": cabin,
                "is_round_trip": is_round_trip,
            }

            state_dict["altered_flight_info"] = flight_info

    # -----------#
    #   Return   #
    # -----------#
    return flight_info, assistant_response


##################################
#   Get Flight Status from API   #
##################################
def get_flight_status(flight_ident):
    """
    Requests flight status for a specific flight using today's date in UTC.
    Args:
        flight_ident (str): The flight identifier (e.g., 'WN1905').
    Returns:
        flight_data (dict): A dictionary containing flight status information.
    """
    api_key = os.environ.get("AEROAPI_KEY")
    base_url = "https://aeroapi.flightaware.com/aeroapi/"

    # Get today's date in UTC
    now_utc = datetime.now(pytz.utc)
    # Go back 2 days UTC for long international flights
    start_date = (now_utc - timedelta(days=2)).strftime("%Y-%m-%d")
    # With no end_date the API defaults to its max of 2 days after today's date

    # Define headers for authentication
    headers = {"x-apikey": api_key}

    # Define parameters for the API request
    params = {
        "ident_type": "designator",
        "start": start_date,
    }

    url = base_url + "flights/" + flight_ident

    # Make the request
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        flight_data = response.json()
    else:
        logging.error(
            f"Error fetching flight status: {response.status_code} - {response.text}"
        )
        flight_data = None

    return flight_data


####################################
#   Get Flights to Book from API   #
####################################
def get_flights_to_book(flight_info, passengers, state_dict):
    """
    Requests flight listings for 1 to 6 origin-destination pairs with dates.
    Args:
        flight_info (dict): A dictionary containing flight information.
        passengers (dict): A dictionary containing passenger information.
        state_dict (dict): A dictionary containing state-related information.
    Returns:
        flight_data (dict): A dictionary containing flight listings.
    """
    LIMIT = "50"
    expedia_key = os.environ.get("EXPEDIA_KEY")
    expedia_auth = os.environ.get("EXPEDIA_AUTH")
    my_company_name = "ActualIntelligence" + "-"
    chat_number = 1  # We may increment this later using sqlite
    chat_number_str = str(chat_number) + "-"
    transaction_id = str(uuid.uuid4())
    partner_transaction_id = my_company_name + chat_number_str + transaction_id
    request_url = "https://apim.expedia.com/flights/listings"

    # Define headers for authentication
    headers = {
        "Accept": "application/vnd.exp-flight.v3+json",
        "Partner-Transaction-Id": partner_transaction_id,
        "Key": expedia_key,
        "Authorization": expedia_auth,
    }

    # Define parameters for the API request
    params = {}

    """ add check for 6 passengers max later """
    if passengers["adult"] > 0:
        params["adult"] = str(passengers["adult"])
    if passengers["senior"] > 0:
        params["senior"] = str(passengers["senior"])
    if passengers["childrenAges"][0] > 0:
        params["childrenAges"] = str(passengers["childrenAges"])
    if passengers["infantInLap"] > 0:
        params["infantInLap"] = str(passengers["infantInLap"])
    if passengers["infantInSeat"] > 0:
        params["infantInSeat"] = str(passengers["infantInSeat"])

    params["cabinClass"] = flight_info["cabin"]

    params["limit"] = LIMIT

    # Restrict search results to the airport code provided in each origin/destination
    # parameter. This filter is ignored if a city name is used instead of an airport code.
    params["filterNearByAirport"] = True

    include_nearby_airports = state_dict["include_nearby_airports"]

    # Build flight segments
    for i, leg in enumerate(flight_info["legs"]):

        if (
            leg["origin"]["use_all_airports"]
            or (state_dict["all_airports"] and len(leg["origin"]["airport"]) > 1)
            or include_nearby_airports
        ):
            # Use city, state for multi-airport city
            origin = leg["origin"]["city"] + ", " + leg["origin"]["airport"][0]["state"]
        else:
            # Use airport code
            origin = leg["origin"]["airport"][0]["code"]

        if (
            leg["destination"]["use_all_airports"]
            or (state_dict["all_airports"] and len(leg["destination"]["airport"]) > 1)
            or include_nearby_airports
        ):
            # Use city, state for multi-airport city
            destination = (
                leg["destination"]["city"]
                + ", "
                + leg["destination"]["airport"][0]["state"]
            )
        else:
            # Use airport code
            destination = leg["destination"]["airport"][0]["code"]

        segment = "segment" + str(i + 1)
        params[segment + ".origin"] = origin
        params[segment + ".destination"] = destination
        params[segment + ".departureDate"] = leg["date"].strftime("%Y-%m-%d")

    # Make the request
    response = requests.get(request_url, headers=headers, params=params)

    if response.status_code == 200:
        flight_data = response.json()
    else:
        flight_data = None
        response_data = response.json()
        logging.error(
            f"Error fetching flights to book: {response.status_code} - {response.text}"
        )
        # Set a default message for unexpected cases
        state_dict["expedia_api_error"] = (
            "An unexpected error occurred with the flight search. Please try again."
        )
        try:
            # Attempt to access the error description
            # Accessing nested keys and list indices can raise KeyError or IndexError
            if (
                response_data
                and response_data.get("Errors")
                and response_data["Errors"][0].get("Description")
            ):
                error_description = response_data["Errors"][0]["Description"]
                state_dict["expedia_api_error"] = error_description
            else:
                # Handle cases where the 'Errors' key or description is missing
                # but no exception was raised
                logging.warning(
                    f"Expedia API response structure unexpected or no errors key: {response.text}"
                )
        except (KeyError, IndexError, TypeError) as e:
            # Catch potential errors when accessing nested data
            # (e.g., key missing, index out of range, not a dict/list)
            logging.error(
                f"Unexpected Expedia API error structure: {response.text}. Error: {e}"
            )
        except Exception as e:
            # Catch any other unexpected errors during processing
            logging.error(
                f"An unexpected error occurred while processing Expedia API response: {e}"
            )

    return flight_data


#########################
#   Get Timezone Name   #
#########################
def get_timezone_name(latitude, longitude, tz_finder=timezone_finder):
    """
    Finds the IANA timezone name for a given latitude and longitude using a
    pre-initialized TimezoneFinder instance.
    Args:
        latitude (str or float): The latitude of the location.
        longitude (str or float): The longitude of the location.
        tz_finder (TimezoneFinder): A pre-initialized TimezoneFinder object.
            Defaults to the globally initialized timezone_finder.
    Returns:
        timezone_name (str): The IANA timezone name (e.g., 'America/New_York'),
            or None if not found.
    """
    try:
        # timezonefinder expects floats
        latitude = float(latitude)
        longitude = float(longitude)
        timezone_name = tz_finder.timezone_at(lng=longitude, lat=latitude)

        return timezone_name

    except (ValueError, TypeError) as e:
        logging.error(
            f"Error finding timezone for lat={latitude}, lon={longitude}: {e}"
        )
        return None


##########################
#   Get Local DateTime   #
##########################
def get_local_datetime(datetime_string, local_timezone_str):
    """
    Converts a datetime string (which can include UTC or an offset)
    to a timezone-aware local datetime object.
    Args:
        datetime_string (str): The datetime string to parse (e.g., 'YYYY-MM-DDTHH:MM:SSZ'
            or 'YYYY-MM-DDTHH:MM:SS±HH:MM').
        local_timezone_str (str): The target local timezone string (e.g., 'America/New_York').
    Returns:
        local_datetime (datetime): A timezone-aware local datetime object,
            or None if parsing or timezone conversion fails.
    """
    try:
        # Use dateutil.parser.parse to handle both 'Z' and offset formats
        gmt_datetime_utc = parser.parse(datetime_string)

        # Ensure the parsed datetime is timezone-aware (it should be with dateutil for these formats)
        if gmt_datetime_utc.tzinfo is None:
            # If for some reason it's not timezone-aware, assume UTC if it ends with Z
            if datetime_string.endswith("Z"):
                gmt_datetime_utc = gmt_datetime_utc.replace(tzinfo=pytz.utc)
            else:
                # If no timezone info and not ending in Z, we can't proceed reliably
                logging.warning(
                    f"Warning: Parsed datetime string has no timezone information: {datetime_string}"
                )
                return None

        # Get the local timezone
        try:
            local_timezone = pytz.timezone(local_timezone_str)
        except pytz.UnknownTimeZoneError:
            logging.error(f"Error: Unknown timezone: {local_timezone_str}")
            return None

        # Convert the UTC datetime to the local timezone
        local_datetime = gmt_datetime_utc.astimezone(local_timezone)

        return local_datetime

    except (ValueError, TypeError) as e:
        logging.error(
            f"get_local_datetime: Could not parse datetime string: {datetime_string}. Error: {e}"
        )
        return None


################################
#   Append Departure Details   #
################################
def append_departure_details(flight, response_lines):
    """
    Appends flight departure details to the response lines.
    Args:
        flight (dict): A dictionary containing flight details.
        response_lines (list): A list to store response lines.
    """
    scheduled_departure_local_datetime = get_local_datetime(
        flight["scheduled_out"], flight["origin"]["timezone"]
    )
    scheduled_departure_local_datetime_str = (
        scheduled_departure_local_datetime.strftime("%H:%M %Z %b-%d")
    )

    arrival_time_str = flight["estimated_in"] or flight["scheduled_in"]
    scheduled_arrival_local_datetime = get_local_datetime(
        arrival_time_str, flight["destination"]["timezone"]
    )
    scheduled_arrival_local_datetime_str = scheduled_arrival_local_datetime.strftime(
        "%H:%M %Z %b-%d"
    )

    response_lines.append(
        f"  Origin: {flight['origin']['city']} ({flight['origin']['code_iata']})"
    )
    response_lines.append(
        f"  Destination: {flight['destination']['city']} ({flight['destination']['code_iata']})"
    )
    response_lines.append(
        f"  Scheduled departure: {scheduled_departure_local_datetime_str}"
    )
    response_lines.append(
        f"  Estimated arrival: {scheduled_arrival_local_datetime_str}"
    )

    departure_gate = flight["gate_origin"]
    departure_terminal = flight["terminal_origin"]
    if departure_gate is None and departure_terminal is None:
        response_lines.append(f"  Gate, Terminal (not avail)")
    else:
        if departure_gate is None:
            departure_gate = "(not avail)"
        if departure_terminal is None:
            departure_terminal = "(not avail)"
        response_lines.append(f"  Gate {departure_gate} Terminal {departure_terminal}")


##############################
#   Append Arrival Details   #
##############################
def append_arrival_details(flight, response_lines):
    """
    Appends flight arrival details to the response lines.
    Args:
        flight (dict): A dictionary containing flight details.
        response_lines (list): A list to store response lines.
    """
    scheduled_arrival_local_datetime = get_local_datetime(
        flight["scheduled_in"], flight["destination"]["timezone"]
    )
    scheduled_arrival_local_datetime_str = scheduled_arrival_local_datetime.strftime(
        "%H:%M %Z %b-%d"
    )

    response_lines.append(
        f"  Origin: {flight['origin']['city']} ({flight['origin']['code_iata']})"
    )
    response_lines.append(
        f"  Destination: {flight['destination']['city']} ({flight['destination']['code_iata']})"
    )
    response_lines.append(
        f"  Scheduled arrival: {scheduled_arrival_local_datetime_str}"
    )


#########################
#   Format Delay Time   #
#########################
def format_delay_time(delay_seconds):
    """
    Converts a delay time in seconds to a human-readable format.
    Args:
        delay_seconds (int): The delay time in seconds.
    Returns:
        str: A string representing the delay time in a human-readable format.
    """
    # Use the absolute value for divmod
    hours, remainder = divmod(abs(delay_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    hours_text = f"{hours} hour{'s' if hours != 1 else ''}"
    minutes_text = f"{minutes} minute{'s' if minutes != 1 else ''}"
    seconds_text = f"{seconds} second{'s' if seconds != 1 else ''}"

    if hours != 0:
        if minutes != 0:
            return hours_text + " " + minutes_text
        else:
            return hours_text
    elif minutes != 0:
        # Unexpected, delay seconds seem to always be multiples of 60
        if seconds != 0:
            return minutes_text + " " + seconds_text
        else:
            return minutes_text
    else:
        # Unexpected, delay seconds seem to always be multiples of 60
        return "less than a minute"


#############################
#   Build Status Response   #
#############################
def build_status_response(flight_data, flight_ident):
    """
    Builds a string containing flight status for a specific flight.
    Filters flights based on arrival date in the destination timezone.
    Args:
        flight_data (dict): A dictionary containing flight status information.
        flight_ident (str): The flight identifier (e.g., 'WN1905').
    Returns:
        str: A string containing the formatted flight status, or a message if no data.
    """
    response_lines = []  # List to build the response string
    relevant_flights = []  # List to store flights relevant to the arrival date

    if flight_data and flight_data.get("flights"):
        # Flight data is returned newest to oldest
        diverted_flight = False
        airline_code = flight_ident[:2]
        flight_number = flight_ident[2:]
        # Defaults to airline code if name isn't found
        airline_name = carrier_codes.get(airline_code, airline_code)
        flight_ident_text = airline_name + " flight " + flight_number

        now_utc = datetime.now(pytz.utc)  # Get today's datetime in UTC

        destination_timezone_str = flight_data["flights"][0]["destination"]["timezone"]
        destination_timezone = pytz.timezone(destination_timezone_str)
        now_local_dest_datetime = now_utc.astimezone(
            destination_timezone
        )  # Get today's local datetime of destination
        target_arrival_date = now_local_dest_datetime.date()
        target_arrival_date_plus_one = target_arrival_date + timedelta(days=1)

        origin_timezone_str = flight_data["flights"][0]["origin"]["timezone"]
        origin_timezone = pytz.timezone(origin_timezone_str)
        now_local_origin_datetime = now_utc.astimezone(
            origin_timezone
        )  # Get today's local datetime of origin

        # Filter flights based on arrival date in the destination timezone
        for flight in flight_data["flights"]:

            # Parse the relevant arrival time string (actual > estimated > scheduled)
            arrival_time_str = (
                flight["actual_in"] or flight["estimated_in"] or flight["scheduled_in"]
            )

            if arrival_time_str:
                arrival_datetime_local = get_local_datetime(
                    arrival_time_str, destination_timezone_str
                )
                arrival_date_local = arrival_datetime_local.date()

                # Check if the arrival date is today or the next day
                # in the destination timezone
                if (
                    arrival_date_local == target_arrival_date
                    or arrival_date_local == target_arrival_date_plus_one
                ):
                    flight["arrival_datetime_local"] = (
                        arrival_datetime_local  # Save for later
                    )
                    relevant_flights.append(flight)

        # Check relevant flights for status
        relevant_flights_length = len(relevant_flights)

        for i in range(relevant_flights_length):
            flight = relevant_flights[i]

            if relevant_flights_length > 1 and i + 1 < relevant_flights_length:
                # Each next flight in the list is an earlier datetime than the last
                next_flight = relevant_flights[i + 1]

                if next_flight["diverted"]:
                    diverted_flight = True
                    response_lines.append(f"{flight_ident_text} is diverted.\n")
                    append_arrival_details(next_flight, response_lines)

                    if flight["fa_flight_id"] != next_flight["fa_flight_id"]:
                        response_lines.append(
                            "\nNo diverted flight data available to display."
                        )
                        break

                else:
                    # Process the flight closest to the scheduled local arrival
                    # or departure time compared to now's time
                    arrival_time_difference = abs(
                        now_local_dest_datetime - next_flight["arrival_datetime_local"]
                    )

                    # Parse the relevant departure time string (actual > estimated > scheduled)
                    departure_time_str = (
                        flight["actual_out"]
                        or flight["estimated_out"]
                        or flight["scheduled_out"]
                    )
                    departure_datetime_local = get_local_datetime(
                        departure_time_str, origin_timezone_str
                    )
                    departure_time_difference = abs(
                        now_local_origin_datetime - departure_datetime_local
                    )

                    # Is the earlier flight's arrival closer to 'now' than
                    # the later flight's departure?
                    if arrival_time_difference < departure_time_difference:
                        flight = next_flight

            if flight["cancelled"]:
                response_lines.append(f"{flight_ident_text} is cancelled.\n")
                append_arrival_details(flight, response_lines)
                scheduled_departure_local_datetime = get_local_datetime(
                    flight["scheduled_out"], flight["origin"]["timezone"]
                )
                scheduled_departure_local_datetime_str = (
                    scheduled_departure_local_datetime.strftime("%H:%M %Z %b-%d")
                )
                response_lines.insert(
                    -1,
                    f"  Scheduled departure: {scheduled_departure_local_datetime_str}",
                )
                break

            # Flight hasn't departed yet
            if flight["actual_out"] is None:
                departure_time_str = flight["estimated_out"] or flight["scheduled_out"]
                departure_datetime_local = get_local_datetime(
                    departure_time_str, origin_timezone_str
                )
                local_departure_datetime_str = departure_datetime_local.strftime(
                    "%H:%M %Z %b-%d"
                )
                local_departure_time = departure_datetime_local.strftime("%H:%M %Z")
                departure_delay_seconds = flight["departure_delay"]

                if departure_delay_seconds == 0:
                    response_lines.append(
                        f"{flight_ident_text} estimated departure on time at {local_departure_time}.\n"
                    )
                else:
                    delay_time_text = format_delay_time(departure_delay_seconds)

                    if departure_delay_seconds > 0:
                        response_lines.append(
                            f"{flight_ident_text} estimated departure {delay_time_text} late at {local_departure_time}.\n"
                        )
                    else:
                        response_lines.append(
                            f"{flight_ident_text} estimated departure {delay_time_text} early at {local_departure_time}.\n"
                        )

                append_departure_details(flight, response_lines)
                break

            # Flight has departed
            if flight["actual_in"] is not None:
                actual_or_estimated_text = "arrived"
            else:
                actual_or_estimated_text = "estimated arrival"

            local_datetime = flight["arrival_datetime_local"]
            local_arrival_datetime_str = local_datetime.strftime("%H:%M %Z %b-%d")
            local_arrival_time = local_datetime.strftime("%H:%M %Z")
            arrival_delay_seconds = flight["arrival_delay"]

            if diverted_flight:
                response_lines.append("\nThe diverted flight data follows:\n")

            if arrival_delay_seconds == 0:
                response_lines.append(
                    f"{flight_ident_text} {actual_or_estimated_text} on time at {local_arrival_time}.\n"
                )
            else:
                delay_time_text = format_delay_time(arrival_delay_seconds)

                if arrival_delay_seconds > 0:
                    response_lines.append(
                        f"{flight_ident_text} {actual_or_estimated_text} {delay_time_text} late at {local_arrival_time}.\n"
                    )
                else:
                    response_lines.append(
                        f"{flight_ident_text} {actual_or_estimated_text} {delay_time_text} early at {local_arrival_time}.\n"
                    )

            append_arrival_details(flight, response_lines)

            if actual_or_estimated_text == "arrived":
                response_lines.append(
                    f"  Actual arrival:    {local_arrival_datetime_str}"
                )
            else:
                response_lines.append(
                    f"  Estimated arrival: {local_arrival_datetime_str}"
                )

            arrival_gate = flight["gate_destination"]
            arrival_terminal = flight["terminal_destination"]
            if arrival_gate is None and arrival_terminal is None:
                response_lines.append(f"  Gate, Terminal (not avail)")
            else:
                if arrival_gate is None:
                    arrival_gate = "(not avail)"
                if arrival_terminal is None:
                    arrival_terminal = "(not avail)"
                response_lines.append(
                    f"  Gate {arrival_gate} Terminal {arrival_terminal}"
                )

            baggage_claim = flight["baggage_claim"]
            if baggage_claim is None:
                response_lines.append(f"  Baggage Claim (not avail)")
            else:
                response_lines.append(f"  Baggage Claim {baggage_claim}")

            break

        # End of relevant_flights for loop

    if response_lines == []:
        response_lines.append("No flight data available to display.")

    return "\n".join(response_lines)  # Join the lines into a single string


##############################
#   Build Booking Response   #
##############################
def build_booking_response(booking_data, flight_info, tz_finder):
    """
    Builds a string containing booking details.
    Args:
        booking_data (dict): A dictionary containing booking data.
        flight_info (dict): A dictionary containing flight information.
        tz_finder (TimezoneFinder): An instance of TimezoneFinder for timezone lookup.
    Returns:
        header_string (str): A string containing the formatted booking header.
        flight_options_list (list): A list of dictionaries containing flight options.
    """
    # Use a list of dictionaries to save response lines and sorting keys for each Offer
    flight_options_list = []

    if booking_data and "Offers" in booking_data:
        # Build header line
        leg_origin = flight_info["legs"][0]["origin"]
        leg_destination = flight_info["legs"][0]["destination"]
        origin_city = leg_origin["city"]
        dest_city = leg_destination["city"]
        if origin_city[-1].isupper():
            origin_city_state = origin_city
        else:
            origin_city_state = origin_city + ", " + leg_origin["airport"][0]["state"]

        if dest_city[-1].isupper():
            dest_city_state = dest_city
        else:
            dest_city_state = dest_city + ", " + leg_destination["airport"][0]["state"]
        departure_date = flight_info["legs"][0]["date"]
        departure_date_str = departure_date.strftime("%b %d, %Y")
        num_legs = len(flight_info["legs"])
        if num_legs == 1:
            header_string = f"{origin_city_state} to {dest_city_state} &nbsp;|&nbsp; {departure_date_str}"
        else:
            return_date = flight_info["legs"][-1]["date"]
            return_date_str = return_date.strftime("%b %d, %Y")
            if (
                num_legs == 2
                and origin_city == flight_info["legs"][1]["destination"]["city"]
            ):
                header_string = (
                    f"{origin_city_state} to {dest_city_state} (round trip) &nbsp;|&nbsp; "
                    + f"{departure_date_str} - {return_date_str}"
                )
            else:
                leg_final_dest = flight_info["legs"][-1]["destination"]
                final_dest = leg_final_dest["city"]
                if final_dest[-1].isupper():
                    final_dest_state = final_dest
                else:
                    final_dest_state = (
                        final_dest + ", " + leg_final_dest["airport"][0]["state"]
                    )
                if num_legs == 2:
                    header_string = (
                        f"{origin_city_state}; {dest_city_state}; {final_dest_state} "
                        + f"({num_legs} legs) &nbsp;|&nbsp; {departure_date_str} - {return_date_str}"
                    )
                else:
                    header_string = (
                        f"{origin_city_state}; {dest_city_state}; ... {final_dest_state} "
                        + f"({num_legs} legs) &nbsp;|&nbsp; {departure_date_str} - {return_date_str}"
                    )

        # Build flight options list
        for i, offer in enumerate(booking_data["Offers"]):
            response_lines = []  # Use a list to build the response string
            segment_ids_list = offer["SegmentIds"]
            first_segment = True

            for segment_id in segment_ids_list:
                for segment in booking_data["Segments"]:
                    if segment["SegmentId"] == segment_id:
                        leg_index = len(segment["Legs"]) - 1

                        # Get leg data
                        for leg in segment["Legs"]:
                            origin_city = leg["DepartureAirport"]["City"]
                            origin_code = leg["DepartureAirport"]["Code"]
                            dest_city = leg["ArrivalAirport"]["City"]
                            dest_code = leg["ArrivalAirport"]["Code"]
                            origin = f"{origin_city} ({origin_code})"
                            destination = f"{dest_city} ({dest_code})"
                            response_lines.append(
                                f"  {leg['MarketingAirlineName']} ({leg['MarketingAirlineCode']}"
                                + f" {leg['FlightNumber']} {origin} - {destination})"
                            )

                        # Get origin timezone once per offer/segment
                        origin_timezone = get_timezone_name(
                            segment["Legs"][0]["DepartureAirport"]["Latitude"],
                            segment["Legs"][0]["DepartureAirport"]["Longitude"],
                            tz_finder,
                        )
                        # Fallback to UTC if timezone not found
                        if origin_timezone is None:
                            origin_timezone = "UTC"
                        local_departure_datetime = get_local_datetime(
                            segment["Legs"][0]["DepartureDateTime"], origin_timezone
                        )
                        # Format the datetime object into a 12-hour string with AM/PM
                        local_departure_time = local_departure_datetime.strftime(
                            "%I:%M %p"
                        )
                        # Extract the time object to use in sorting later
                        if first_segment:
                            first_segment = False
                            time_for_sorting = local_departure_datetime.time()

                        # Get destination timezone once per offer/segment
                        destination_timezone = get_timezone_name(
                            segment["Legs"][leg_index]["ArrivalAirport"]["Latitude"],
                            segment["Legs"][leg_index]["ArrivalAirport"]["Longitude"],
                            tz_finder,
                        )
                        # Fallback to UTC if timezone not found
                        if destination_timezone is None:
                            destination_timezone = "UTC"
                        local_arrival_datetime = get_local_datetime(
                            segment["Legs"][leg_index]["ArrivalDateTime"],
                            destination_timezone,
                        )
                        # Format the datetime object into a 12-hour string with AM/PM
                        local_arrival_time = local_arrival_datetime.strftime("%I:%M %p")

                        duration_timedelta = isodate.parse_duration(
                            segment["FlightDuration"]
                        )
                        total_seconds = duration_timedelta.total_seconds()
                        hours, remainder = divmod(total_seconds, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        duration_text = f"{int(hours)}h {int(minutes)}m"

                        total_stops = segment["TotalStops"]
                        if total_stops == 0:
                            total_stops_text = "nonstop"
                        else:
                            total_stops_text = (
                                f"{total_stops} stop{'s' if total_stops > 1 else ''}"
                            )

                        response_lines.append(
                            f"  {local_departure_time} - {local_arrival_time} ({duration_text},"
                            + f" {total_stops_text})"
                        )

            total_price = offer["OfferPrice"]["TotalPrice"][
                "Value"
            ]  # Access the Value key
            currency = offer["OfferPrice"]["TotalPrice"][
                "Currency"
            ]  # Access the Currency key
            url = offer["Links"]["WebDetails"]["Href"]
            link_text = "View Details and Book"

            if currency == "USD":
                # Use HTML anchor tag for link
                response_lines.append(
                    f"  ${total_price}  <a href='{url}'>{link_text}</a>"
                )
            else:
                response_lines.append(
                    f"  {total_price} {currency}  <a href='{url}'>{link_text}</a>"
                )

            inner_dict = {
                "price": total_price,
                "time": time_for_sorting,
                "duration": total_seconds,
            }
            # Join the lines into a single string with HTML line breaks
            inner_dict["display_text"] = "<br>".join(response_lines)
            flight_options_list.append(inner_dict)

    else:
        header_string = "No flights available to display."

    return header_string, flight_options_list


#############################
#   Update Relative Dates   #
#############################
def update_relative_dates(state_dict):
    """
    Updates the relative dates in flight_info.
    Args:
        state_dict (dict): The state dictionary.
    Returns:
        flight_info (dict): The flight information dictionary.
    """
    flight_info = state_dict["altered_flight_info"]
    date_dict = state_dict["date_dict"]

    valid_dates, invalid_dates = standardize_date_strings(date_dict, state_dict)

    date_dict["dates"] = valid_dates

    # Replace departure dates in each leg
    for leg, date_obj in zip(flight_info["legs"], date_dict["dates"]):
        leg["date"] = date_obj

    return flight_info


####################################
#   Parse Multi-Airport Response   #
####################################
def parse_multi_airport_response(user_input, state_dict):
    """
    Parses the airport code(s) or line number(s) in the follow-up user response,
    and updates flight_info_alt with the selected airport code(s).
    Args:
        user_input (str): The user's message.
        state_dict (dict): The state dictionary.
    Returns:
        flight_info_alt (dict): The updated flight information dictionary.
        assistant_response (str): The assistant's response.
    """
    assistant_response = ""

    if "cancel" in user_input.lower():
        state_dict["multi_airport_prompt_active"] = False
        del state_dict["original_flight_info"]
        del state_dict["altered_flight_info"]
        assistant_response = "Enter another flight query..."
        return {}, assistant_response

    # Parse the user input and update flight_info_alt with the selected airport code(s).
    multi_airport_display_string = state_dict["multi_airport_display_string"]
    flight_info_alt = state_dict["altered_flight_info"]
    origin_airport_codes = []
    destination_airport_codes = []
    all_airport_codes = []
    errors = []

    # Collect airport codes of multi-airport cities
    leg = flight_info_alt["legs"][0]
    num_origin_airports = len(leg["origin"]["airport"])
    num_destination_airports = len(leg["destination"]["airport"])
    required_selections = 0
    if num_origin_airports > 1:
        required_selections += 1
        origin_airport_codes = [airport["code"] for airport in leg["origin"]["airport"]]
        origin_airport_codes.append("ALL")
        num_origin_airports += 1
        all_airport_codes.extend(origin_airport_codes)
    if num_destination_airports > 1:
        required_selections += 1
        destination_airport_codes = [
            airport["code"] for airport in leg["destination"]["airport"]
        ]
        destination_airport_codes.append("ALL")
        num_destination_airports += 1
        all_airport_codes.extend(destination_airport_codes)

    num_all_airport_codes = len(all_airport_codes)

    if num_all_airport_codes < 10:
        # Check the input for 3-char alpha or 1-char numeric within the airports list range
        # Use a non-capturing group (?:...) for the OR condition, and escape curly braces within the f-string
        under_9_airports_regex = rf"\b(?:[A-Z]{{3}}|[1-{num_all_airport_codes}]{{1}})\b"
        compiled_airports_regex = re.compile(under_9_airports_regex, re.IGNORECASE)
        valid_input = compiled_airports_regex.findall(user_input)

    else:
        # Unexpected, would not expect 10 or more airports in two cities max
        # Check the input for 3-char alpha or 1 to 2-char numeric
        regex_matches = OVER_9_AIRPORTS_REGEX.findall(user_input)
        # Remove invalid 1 to 2-char numerics if any
        valid_input = []
        for item in regex_matches:
            if item.isdigit():
                line_number = int(item)
                if line_number <= num_all_airport_codes and line_number != 0:
                    valid_input.append(item)
            else:
                valid_input.append(item)

    if not valid_input:
        if required_selections == 1:
            errors.append(
                "- Your input didn't include a 3-character airport code or valid line number."
            )
        else:
            errors.append(
                "- Your input didn't include any 3-character airport codes or valid line numbers."
            )

    else:
        origin_airport_match = []
        destination_airport_match = []
        origin_input = []
        destination_input = []

        # Check if each string in valid_input matches an airport code in separate cities
        for item in valid_input:

            if item.isdigit():
                # Item is a line number
                line_number = int(item)

                if num_origin_airports > 1 and line_number <= num_origin_airports:
                    # Line number matches an origin airport
                    origin_airport_match.append(origin_airport_codes[line_number - 1])
                    origin_input.append(line_number)
                else:
                    # Must be a destination line number
                    if required_selections == 1:
                        # Just 1 city with multi-airports
                        dest_index = line_number - 1
                    else:
                        # Both cities have multi-airports
                        dest_index = line_number - num_origin_airports - 1
                    # Save the destination airport code
                    destination_airport_match.append(
                        destination_airport_codes[dest_index]
                    )
                    destination_input.append(line_number)

            else:
                # Item is 3-char alpha
                airport_code = item.upper()

                if airport_code not in all_airport_codes:
                    errors.append(
                        f"- Selected airport '{airport_code}' isn't one of the choices listed."
                    )
                else:
                    # Airport code matches one in multi-airport list
                    if airport_code == "ALL":
                        if num_origin_airports > 1 and num_destination_airports > 1:
                            # Multi-airports in both origin and destination
                            if not origin_airport_match:
                                origin_airport_match.append("ALL")
                                origin_input.append("ALL")
                            else:
                                destination_airport_match.append("ALL")
                                destination_input.append("ALL")
                        elif num_origin_airports > 1:
                            # Multi-airports in origin only
                            origin_airport_match.append("ALL")
                            origin_input.append("ALL")
                        else:
                            # Multi-airports in destination only
                            destination_airport_match.append("ALL")
                            destination_input.append("ALL")
                    elif airport_code in origin_airport_codes:
                        origin_airport_match.append(airport_code)
                        origin_input.append(airport_code)
                    else:
                        # Must be a destination airport
                        destination_airport_match.append(airport_code)
                        destination_input.append(airport_code)

        # End of valid_input for loop

        # Check if we found the correct number of matches
        if num_origin_airports > 1:
            if not origin_airport_match:
                errors.append("- You didn't select an origin airport.")
            elif len(origin_airport_match) > 1:
                errors.append(
                    f"- Too many origin airports were selected: {origin_input}."
                )

        if num_destination_airports > 1:
            if not destination_airport_match:
                errors.append("- You didn't select a destination airport.")
            elif len(destination_airport_match) > 1:
                errors.append(
                    f"- Too many destination airports were selected: {destination_input}."
                )

    if errors:
        # Add error messages and original multi-airports list to response
        if len(errors) > 1:
            header = "\nSorry, your selection had some issues:"
            errors.insert(0, header)
        else:
            first_letter = errors[0][2].lower()
            header = f"\nSorry, {first_letter}{errors[0][3:]}"
            errors.insert(0, header)
            del errors[1]
        # Add a blank line after the last error message
        errors.append("")
        if required_selections == 1:
            errors.append(
                "Please try again using one airport code or line number from the list above."
            )
        else:
            errors.append(
                f"Please try again using one airport code or line number from the "
                + "origin list and one from the destination list."
            )
        errors.append("Or you can type 'cancel' to start a new query.")
        # Use \n in .join for markdown text
        assistant_response = (
            "_" + multi_airport_display_string + "_\n" + "\n".join(errors)
        )

    else:
        # Update flight_info_alt with the selected airport code(s)
        updated_origin_airport = []
        updated_destination_airport = []

        if origin_airport_match:
            if origin_airport_match[0] == "ALL":
                leg["origin"]["use_all_airports"] = True
            else:
                updated_origin_airport = [
                    airport
                    for airport in leg["origin"]["airport"]
                    if origin_airport_match[0] == airport["code"]
                ]

        if destination_airport_match:
            if destination_airport_match[0] == "ALL":
                leg["destination"]["use_all_airports"] = True
            else:
                updated_destination_airport = [
                    airport
                    for airport in leg["destination"]["airport"]
                    if destination_airport_match[0] == airport["code"]
                ]

        # Update the flight_info leg with selected airport(s)
        if origin_airport_match == "ALL":
            leg["origin"]["use_all_airports"] = True
        elif updated_origin_airport:
            leg["origin"]["airport"] = updated_origin_airport
        if destination_airport_match == "ALL":
            leg["destination"]["use_all_airports"] = True
        elif updated_destination_airport:
            leg["destination"]["airport"] = updated_destination_airport

        if flight_info_alt["is_round_trip"]:
            flight_info_alt["legs"][1]["origin"] = leg["destination"]
            flight_info_alt["legs"][1]["destination"] = leg["origin"]

    return flight_info_alt, assistant_response


###########################
#   Sort Flight Options   #
###########################
def sort_flight_options(flight_options, sort_radio):
    """
    Sorts the flight options based on the selected sorting button.
    Args:
        flight_options (list): A list of flight options.
        sort_radio (str): The selected sorting button.
    Returns:
        list: The sorted list of flight options.
    """
    if sort_radio == "Price":
        # Sort by price (ascending) then by time
        return sorted(flight_options, key=lambda x: (float(x["price"]), x["time"]))
    elif sort_radio == "Time":
        # Sort by time (ascending) then by price
        return sorted(flight_options, key=lambda x: (x["time"], float(x["price"])))
    else:
        # Sort by stops
        def get_sort_key(option):
            """Helper function to determine the sorting key."""
            # Extract the number of stops from the display text (duration, X stop/stops)
            stops_match = NUMBER_OF_STOPS_REGEX.search(option["display_text"])
            num_stops = (
                int(stops_match.group(1)) if stops_match else 0
            )  # Default to 0 for nonstops

            if sort_radio == "Price (Nonstops first)":
                # Sort by number of stops (ascending), then price (ascending), then time
                return (num_stops, float(option["price"]), option["time"])

            elif sort_radio == "Time (Nonstops first)":
                # Sort by number of stops (ascending), then time (ascending), then price
                return (num_stops, option["time"], float(option["price"]))

        return sorted(flight_options, key=get_sort_key)


##################################
#   Build Flight Options Batch   #
##################################
def build_flight_options_batch(sorted_options, start_index, batch_size=10):
    """
    Builds a formatted display string for a batch of flight options.
    Args:
        sorted_options (list): The list of sorted flight options.
        start_index (int): The starting index for the batch.
        batch_size (int): The number of options in the batch.
    Returns:
        str: A formatted string containing a batch of flight options,
             or an empty string if no options.
    """
    response_lines = []

    for i in range(batch_size):
        list_index = i + start_index
        if list_index < len(sorted_options):
            response_lines.append(f"<br>--- Option {list_index + 1} ---")
            response_lines.append(f"{sorted_options[list_index]['display_text']}")

    if i + start_index + 1 >= len(sorted_options):
        response_lines.append(f"<br>--- End of flight options ---")
    else:
        response_lines.append(f"<br>--- View more flight options (button on right) ---")

    return "<br>".join(response_lines)


################################
#   Show More Flight Options   #
################################
def show_more_flight_options(chat_history, state_dict):
    """
    Builds the response for showing more flight options.
    Args:
        chat_history (list): The history of the conversation (list of dictionaries).
        state (dict or None): The current state dictionary.
    Returns:
        chat_history (list): The updated chat history (list of dictionaries).
        state_dict (dict): The updated state dictionary.
    """
    response_lines = []

    if "flight_options_batch_n" in state_dict:
        # Get stored sort preference, else default to Price
        current_sort_preference = state_dict.get("active_sort_preference", "Price")
        sort_radio = state_dict["sort_radio"]

        if current_sort_preference != sort_radio:
            # Sorting has changed, reset to the first batch and re-sort the list
            state_dict["flight_options_batch_n"] = 1
            state_dict["active_sort_preference"] = sort_radio

            sorted_options = sort_flight_options(
                state_dict["flight_options_list"], sort_radio
            )
            # Store the newly sorted list
            state_dict["flight_options_list"] = sorted_options

            response_lines.append(
                f"<br>--- Flight Options (sorted by {sort_radio}) ---"
            )
            response_lines.append(state_dict["header_string"])

        else:
            # Sorting hasn't changed, just update the starting index
            state_dict["flight_options_batch_n"] += 1

        next_starting_index = (state_dict["flight_options_batch_n"] - 1) * 10

        if next_starting_index < len(state_dict["flight_options_list"]):
            response_lines.append(
                build_flight_options_batch(
                    state_dict["flight_options_list"], next_starting_index
                )
            )
            # Join the lines into a single string
            assistant_response = "<br>".join(response_lines)
            # Add the assistant's message to chat_history
            chat_history.append({"role": "assistant", "content": assistant_response})

        else:
            # No more options to display
            if (
                chat_history
                and chat_history[-1]["content"] != "<br>No more flights to display."
            ):  # Avoid adding the same message repeatedly
                chat_history.append(
                    {"role": "assistant", "content": "<br>No more flights to display."}
                )

    else:
        # No flight options were stored in state_dict (e.g., original search failed)
        if (
            chat_history
            and chat_history[-1]["content"]
            != "<br>No flight options available to display."
        ):  # Avoid adding the same message repeatedly
            chat_history.append(
                {
                    "role": "assistant",
                    "content": "<br>No flight options available to display.",
                }
            )

    return chat_history, state_dict


####################################
#   Chat Flight Assistant driver   #
####################################
def chat_flight_assistant(chat_history, state_dict, user_input):
    """
    Processes user input for the flight assistant chatbot, handling conversational turns.
    Args:
        chat_history (list): The history of the conversation (list of dictionaries). For example,
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
        state (dict or None): The current state dictionary.
        user_input (str): The user's message.
    Returns:
        chat_history (list): The updated chat history (list of dictionaries).
        state_dict (dict): The updated state dictionary.
        str: An empty string to clear the input textbox.
    """
    logging.info(f"User input: {user_input}")
    # Add the user's message to chat history
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_input})
    assistant_response = ""  # Initialize response string
    is_new_query = True

    if state_dict["multi_airport_prompt_active"]:
        # This is a response to the multi-airport prompt.
        flight_info, assistant_response = parse_multi_airport_response(
            user_input, state_dict
        )
        # Check for error response
        if assistant_response:
            # Add the assistant's message to chat_history
            chat_history.append({"role": "assistant", "content": assistant_response})
            return assistant_response, chat_history, state_dict

        # Multi-airports were selected correctly
        else:
            intent = "booking"
            is_new_query = False

    # ----------------------------------------#
    #   Classify intent and parse the query   #
    # ----------------------------------------#
    if is_new_query:
        state_dict["multi_airport_prompt_active"] = False
        state_dict["all_airports"] = False
        response = question_classifier(user_input)
        intent = response[0]["label"]
        state_dict["intent"] = intent

        flight_info, assistant_response = extract_flight_info(
            user_input, intent, state_dict
        )

        # Check for errors
        if assistant_response:
            # Add the assistant's response to chat_history
            chat_history.append({"role": "assistant", "content": assistant_response})
            return assistant_response, chat_history, state_dict

    # ----------------------#
    #   Status processing   #
    # ----------------------#
    if intent == "status":

        flight_ident = flight_info["airline_code"] + flight_info["flight_number"]
        flight_data = get_flight_status(flight_ident)
        assistant_response = build_status_response(flight_data, flight_ident)

    # -----------------------#
    #   Booking processing   #
    # -----------------------#
    elif intent == "booking":

        response_lines = []  # Use a list to build the response string
        num_legs = len(flight_info["legs"])

        if num_legs > 1 and not flight_info["is_round_trip"]:
            # Ignore multi-airport prompt for multi-leg trips
            state_dict["all_airports"] = True

        # Check multi-airport cities for one-ways and round trips
        leg = flight_info["legs"][0]
        origin_airports = leg["origin"]["airport"]
        destination_airports = leg["destination"]["airport"]

        # Check if origin and/or destination cities have multiple airports
        if (
            not state_dict["multi_airport_prompt_active"]
            and (len(origin_airports) > 1 or len(destination_airports) > 1)
            and not state_dict["include_nearby_airports"]
        ):
            # Save all multi-airports once to retry with different airports
            # later if state_dict["multi_airport_prompt_active"] == False:
            state_dict["original_flight_info"] = copy.deepcopy(flight_info)
            state_dict["multi_airport_prompt_active"] = True

            # Build airport options list
            origin_code = None
            destination_code = None
            first_line_number_dest = 0

            if len(origin_airports) > 1:
                origin_code = origin_airports[0]["code"]
                response_lines.append(
                    f"For origin '{leg['origin']['city']}', multiple airports were found:"
                )
                for i, airport in enumerate(origin_airports):
                    # Display the entire key/value from the airport_codes dict,
                    # e.g. "SPI": "Springfield, IL: Abraham Lincoln Capital"
                    city_airport_str = (
                        airport_codes[airport["code"]] + f" ({airport['code']})"
                    )
                    response_lines.append(f"  {i+1}. {city_airport_str}")

                response_lines.append(f"  {i+2}. All nearby airports (ALL)")
                response_lines.append("")
                first_line_number_dest = len(origin_airports) + 1

            if len(destination_airports) > 1:
                destination_code = destination_airports[0]["code"]
                response_lines.append(
                    f"For destination '{leg['destination']['city']}', "
                    + "multiple airports were found:"
                )
                for i, airport in enumerate(
                    destination_airports, start=first_line_number_dest
                ):
                    # Display the entire key/value from the airport_codes dict,
                    # e.g. "SGF": "Springfield, MO: Springfield-Branson National"
                    city_airport_str = (
                        airport_codes[airport["code"]] + f" ({airport['code']})"
                    )
                    response_lines.append(f"  {i+1}. {city_airport_str}")
                response_lines.append(f"  {i+2}. All nearby airports (ALL)")
                response_lines.append("")

            if origin_code is not None and destination_code is not None:
                response_lines.append(
                    f"Please specify which airports you'd like to use (e.g., '1,"
                    + f"{first_line_number_dest + 1}' or '{origin_code} {destination_code}')."
                )
            elif origin_code is not None:
                response_lines.append(
                    f"Please specify which airport you'd like to use "
                    + f"(e.g., '1' or '{origin_code}')."
                )
            else:
                response_lines.append(
                    f"Please specify which airport you'd like to use "
                    + f"(e.g., '1' or '{destination_code}')."
                )

            # Join the lines into a single string
            state_dict["multi_airport_display_string"] = "<br>".join(response_lines)

            # Add trailer and header first time only to avoid redisplaing them when flights aren't found
            response_lines.append("You can also type 'cancel' to start a new query.")
            response_lines.insert(0, "Multiple airports exist for your search.\n")

        # Get booking data
        elif origin_airports and destination_airports:
            origin_code = origin_airports[0]["code"]
            destination_code = destination_airports[0]["code"]
            passengers = {}
            passengers["adult"] = 1
            passengers["senior"] = 0
            passengers["childrenAges"] = [0]
            passengers["infantInLap"] = 0
            passengers["infantInSeat"] = 0

            booking_data = get_flights_to_book(flight_info, passengers, state_dict)

            header_str, options_list = build_booking_response(
                booking_data, flight_info, timezone_finder
            )

            # No flights were found
            if "No flights" in header_str:
                if "expedia_api_error" in state_dict:
                    response_lines.append(
                        f"Flight search failed: {state_dict['expedia_api_error']}"
                    )
                elif state_dict["multi_airport_prompt_active"] == True:
                    # This was after a multi-airport selection
                    original_flight_info = state_dict["original_flight_info"]
                    origin_airports = original_flight_info["origin"]["airport"]
                    destination_airports = original_flight_info["destination"][
                        "airport"
                    ]
                    num_multi_airport_cities = 0
                    if len(origin_airports) > 1:
                        num_multi_airport_cities += 1
                    if len(destination_airports) > 1:
                        num_multi_airport_cities += 1
                    # Add follow-up display lines
                    response_lines.append(
                        "_" + state_dict["multi_airport_display_string"] + "_"
                    )
                    response_lines.append(
                        f"\nNo flights were found for the requested airports "
                        + f"({origin_code}-{destination_code}).\n"
                    )
                    if num_multi_airport_cities > 1:
                        response_lines.append(
                            "You can try again with different airports from the list above."
                        )
                    else:
                        response_lines.append(
                            "If you'd like to try again, select a different airport "
                            + "from the list above."
                        )
                    response_lines.append(
                        "Otherwise type 'cancel' to start a new query."
                    )
                    state_dict["altered_flight_info"] = copy.deepcopy(
                        state_dict["original_flight_info"]
                    )
                else:
                    # Not after a multi-airport selection
                    if num_legs == 1:
                        header_str = (
                            f"No flights were found for the requested airports "
                            + f"({origin_code}-{destination_code})."
                        )
                    elif num_legs == 2:
                        leg_2 = flight_info["legs"][1]
                        dest_code_2 = leg_2["destination"]["airport"][0]["code"]
                        header_str = (
                            f"No flights were found for the requested airports "
                            + f"({origin_code}-{destination_code}-{dest_code_2})."
                        )
                    else:
                        header_str = (
                            "No flights were found for the requested itinerary. "
                            + "Please check your locations and dates and try again."
                        )
                    response_lines.append(header_str)

            # Flights were found
            else:
                # Sort options based on the selected button
                sort_radio = state_dict["sort_radio"]
                sorted_options = sort_flight_options(options_list, sort_radio)

                # Store the booking info in state_dict for follow-up user requests
                state_dict["multi_airport_prompt_active"] = False
                state_dict["active_sort_preference"] = sort_radio
                state_dict["flight_options_batch_n"] = 1
                state_dict["header_string"] = header_str
                state_dict["flight_options_list"] = sorted_options

                # Build the response string for the first 10 options
                response_lines.append(
                    f"--- Flight Options (sorted by {sort_radio}) ---"
                )
                response_lines.append(header_str)
                response_lines.append(build_flight_options_batch(sorted_options, 0))

        # Join the lines into a single string
        assistant_response = "<br>".join(response_lines)

    # Add the assistant's message to chat_history
    chat_history.append({"role": "assistant", "content": assistant_response})

    return assistant_response, chat_history, state_dict


####################
#   Flask Routes   #
####################


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.json
    user_input = data.get("input", "")
    sort_radio = data.get("sort_radio", "Price")
    date_format = data.get("date_format", "MM/DD: 6/23 (U.S.A.)")
    nearby_airports = data.get("nearby_airports", False)
    chat_history = data.get("chat_history", [])
    state_dict = data.get("state_dict", {})

    # Initialize if empty/missing
    if not state_dict:
        state_dict = {
            "multi_airport_prompt_active": False,
            "user_date": None,
            "user_timezone": "Etc/UTC",
        }

    # Update state with current client selections
    state_dict["sort_radio"] = sort_radio
    state_dict["include_nearby_airports"] = nearby_airports
    state_dict["preferred_date_format"] = "MM/DD" if "MM/DD" in date_format else "DD/MM"

    # Extract date/timezone if provided
    date_timezone = data.get("date_timezone", {})
    if date_timezone:
        user_date_str = date_timezone.get("date")
        user_timezone = date_timezone.get("timezone", "Etc/UTC")

        # Parse the local date string into a date object (ignore time for now)
        if user_date_str and state_dict.get("user_date") is None:
            try:
                parsed_date = parser.parse(user_date_str).date()
                state_dict["user_date"] = parsed_date
                state_dict["user_timezone"] = user_timezone  # Save for future use
            except Exception as e:
                logging.warning(
                    f"Failed to parse client date: {user_date_str} | Error: {e}"
                )

    # Call chat_flight_assistant (updated to return assistant_response first)
    assistant_response, chat_history, state_dict = chat_flight_assistant(
        chat_history, state_dict, user_input
    )
    return jsonify(
        {
            "response": assistant_response,
            "chat_history": chat_history,
            "state_dict": state_dict,
        }
    )


@app.route("/show_more", methods=["POST"])
def show_more_endpoint():
    data = request.json
    chat_history = data.get("chat_history", [])
    state_dict = data.get("state_dict", {})
    sort_radio = data.get("sort_radio", "Price")

    # Apply new sort if user changed it
    state_dict["sort_radio"] = sort_radio

    # Call show_more_flight_options
    chat_history, state_dict = show_more_flight_options(chat_history, state_dict)
    assistant_response = (
        chat_history[-1]["content"] if chat_history else "No more flights to display."
    )
    return jsonify(
        {
            "response": assistant_response,
            "chat_history": chat_history,
            "state_dict": state_dict,
        }
    )


@app.route("/")
def index():
    with open("index.html", "r") as f:
        return f.read()


if __name__ == "__main__":
    logging.info("Starting Flask server")
    app.run(host="0.0.0.0", port=7860)
