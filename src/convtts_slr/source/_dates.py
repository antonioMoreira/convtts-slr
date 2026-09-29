from whenever import Date, OffsetDateTime, PlainDateTime


def parse_date(value: str | None) -> Date | None:
    """The calendar date in a backend's metadata, or None when it is missing, partial
    (`"2022"`) or malformed: one bad date must not fail a whole search, and the paper's
    `year` is carried separately.

    Accepts a plain ISO date or an ISO timestamp, with or without an offset. A timestamp
    keeps the date as written (`2023-04-15T23:00:00-05:00` is 2023-04-15), not its UTC date.
    """
    if not value:
        return None
    for parse in (Date.parse_iso, OffsetDateTime.parse_iso, PlainDateTime.parse_iso):
        try:
            parsed = parse(value)
        except ValueError:
            continue
        return parsed if isinstance(parsed, Date) else parsed.date()
    return None
