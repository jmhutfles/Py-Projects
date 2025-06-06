from datetime import datetime, timedelta


#Feet to meters
def FeetToMeters (feet):
    meters = feet / 3.28284
    return meters

#meteres to feet
def MetersToFeet(meters):
    feet = meters * 3.28084
    return feet



def iso_to_gps_week_seconds(iso_str):
    # Parse ISO time
    dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    
    # Get the most recent GPS week start (Sunday 00:00 UTC)
    week_start = dt - timedelta(days=dt.weekday() + 1 if dt.weekday() < 6 else 0,
                                hours=dt.hour,
                                minutes=dt.minute,
                                seconds=dt.second,
                                microseconds=dt.microsecond)
    
    # Seconds since GPS week start
    gps_week_seconds = (dt - week_start).total_seconds()
    
    return int(gps_week_seconds), gps_week_seconds  # rounded, and exact