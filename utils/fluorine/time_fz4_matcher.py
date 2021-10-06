import re

BIN_FREEZING_RATE_HEADER_MATCHER = '^Animal\sID\tGroup\sName\t(bin\d\t)?'
BIN_FREEZING_RATE_ROW_MATCHER = '^\d{1,2}\t(.+?)\t[\d\.]+?\t'


def is_freezing_rate_header(s):
    return re.match(BIN_FREEZING_RATE_HEADER_MATCHER, s)


def is_freezing_rate_row(s):
    return re.match(BIN_FREEZING_RATE_ROW_MATCHER, s)
