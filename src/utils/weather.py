import carla

def get_weather(weather_name):
    weathers = {
        "clear_day": carla.WeatherParameters.ClearNoon,
        "cloudy_day": carla.WeatherParameters.CloudyNoon,
        "wet_day": carla.WeatherParameters.WetNoon,
        "rain_day": carla.WeatherParameters.HardRainNoon,
        "clear_sunset": carla.WeatherParameters.ClearSunset,
        "soft_rain_sunset": carla.WeatherParameters.SoftRainSunset,
    }

    if weather_name not in weathers:
        raise ValueError(f"Unknown weather: {weather_name}")

    return weathers[weather_name]