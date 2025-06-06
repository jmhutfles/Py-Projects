from datetime import datetime, timedelta


#Feet to meters
def FeetToMeters (feet):
    meters = feet / 3.28284
    return meters

#meteres to feet
def MetersToFeet(meters):
    feet = meters * 3.28084
    return feet

def gps_to_utc(gps_week, gps_seconds):
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    leap_seconds = 18  # Update if leap seconds change
    return gps_epoch + timedelta(weeks=int(gps_week), seconds=float(gps_seconds) - leap_seconds)

def convert_sensor_time_to_utc(df):
    # Find the first row with both 'Time (s)', 'tow (s)', and 'week' filled (from $TIME row)
    ref_row = df.dropna(subset=["Time (s)", "tow (s)", "week"]).iloc[0]
    t_sensor_ref = float(ref_row["Time (s)"])
    t_gps_ref = float(ref_row["tow (s)"])
    gps_week = int(ref_row["week"])
    # Compute UTC for each row
    def row_to_utc(t_sensor):
        t_gps = t_gps_ref + (t_sensor - t_sensor_ref)
        return gps_to_utc(gps_week, t_gps)
    df["UTC"] = df["Time (s)"].apply(row_to_utc)
    return df